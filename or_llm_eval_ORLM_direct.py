import openai
import anthropic
from dotenv import load_dotenv
import os
import json
import asyncio
import argparse
import time
from itertools import zip_longest
from utils import (
    is_number_string,
    convert_to_number,
    eval_model_result
)

# Load environment variables from .env file
load_dotenv()

# OpenAI API setup
openai_api_data = dict(
    api_key = os.getenv("OPENAI_API_KEY"),
    base_url = os.getenv("OPENAI_API_BASE")
)

# Anthropic API setup
anthropic_api_data = dict(
    api_key = os.getenv("CLAUDE_API_KEY"),
)

# Initialize clients
openai_client = openai.AsyncOpenAI(
    api_key=openai_api_data['api_key'],
    base_url=openai_api_data['base_url'] if openai_api_data['base_url'] else None
)

anthropic_client = anthropic.AsyncAnthropic(
    api_key=anthropic_api_data['api_key']
)

async def async_query_llm(messages, model_name="o3-mini", temperature=0.2):
    """
    Async version of query_llm that supports both OpenAI and Claude models
    """
    # Check if model is Claude (Anthropic)
    if model_name.lower().startswith("claude"):
        # Convert OpenAI message format to Anthropic format
        system_message = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        assistant_messages = [m["content"] for m in messages if m["role"] == "assistant"]
        
        # Combine messages into a single conversation string
        conversation = system_message + "\n\n"
        for user_msg, asst_msg in zip_longest(user_messages, assistant_messages, fillvalue=None):
            if user_msg:
                conversation += f"Human: {user_msg}\n\n"
            if asst_msg:
                conversation += f"Assistant: {asst_msg}\n\n"
        
        # Add the final user message if there is one
        if len(user_messages) > len(assistant_messages):
            conversation += f"Human: {user_messages[-1]}\n\n"

        response = await anthropic_client.messages.create(
            model=model_name,
            max_tokens=8192,
            temperature=temperature,
            messages=[{
                "role": "user",
                "content": conversation
            }]
        )
        return response.content[0].text
    else:
        # Use OpenAI API
        response = await openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content

def extract_numerical_answer(text):
    """
    Extract numerical answer from LLM response using JSON format.
    Looks for ```json ... ``` blocks containing answer in JSON format.
    """
    import re
    import json
    
    # Look for JSON blocks first (most reliable)
    json_blocks = re.findall(r'```json\s*([\s\S]*?)```', text)
    
    if json_blocks:
        for json_block in json_blocks:
            json_block = json_block.strip()
            try:
                data = json.loads(json_block)
                # Try different possible keys for the answer
                possible_keys = ['answer', 'optimal_value', 'result', 'value', 'solution']
                for key in possible_keys:
                    if key in data:
                        value = data[key]
                        # Handle case where value might be in array format
                        if isinstance(value, list) and len(value) > 0:
                            value = value[0]
                        return str(float(value))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    
    # Fallback: look for any JSON-like structure in the text
    json_patterns = [
        r'\{\s*["\']answer["\']\s*:\s*([+-]?\d+(?:\.\d+)?)\s*\}',
        r'\{\s*["\']optimal_value["\']\s*:\s*([+-]?\d+(?:\.\d+)?)\s*\}',
        r'\{\s*["\']result["\']\s*:\s*([+-]?\d+(?:\.\d+)?)\s*\}',
        r'\{\s*["\']value["\']\s*:\s*([+-]?\d+(?:\.\d+)?)\s*\}',
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                return str(float(matches[0]))
            except ValueError:
                continue
    
    # Final fallback: look for explicit answer patterns
    fallback_patterns = [
        r'答案是[:：]\s*([+-]?\d+(?:\.\d+)?)',
        r'最优解是[:：]\s*([+-]?\d+(?:\.\d+)?)',
        r'最优值是[:：]\s*([+-]?\d+(?:\.\d+)?)',
        r'答案[:：]\s*([+-]?\d+(?:\.\d+)?)',
        r'最优解[:：]\s*([+-]?\d+(?:\.\d+)?)',
        r'最优值[:：]\s*([+-]?\d+(?:\.\d+)?)',
    ]
    
    for pattern in fallback_patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                return str(float(matches[0]))
            except ValueError:
                continue
    
    return None

async def async_direct_answer_solver(user_question, model_name="o3-mini"):
    """
    Directly ask LLM for the answer to an optimization problem without code generation.
    """
    messages = [
        {"role": "system", "content": (
            "你是一个运筹优化专家。请直接分析并求解用户提供的运筹优化问题，给出最优解的数值。"
            "请仔细分析问题，建立数学模型，并计算出最优解。"
            "在回答的最后，请以JSON格式给出最终答案: ```json\n{\"answer\": [数值]}\n```，其中[数值]是最优目标函数值。"
            "只需要给出最优目标函数值，不需要给出具体的变量取值。"
        )},
        {"role": "user", "content": user_question}
    ]

    response = await async_query_llm(messages, model_name)
    print("【LLM直接回答】:\n", response)
    
    # Extract numerical answer from response
    numerical_answer = extract_numerical_answer(response)
    
    if numerical_answer is not None:
        print(f"提取到的数值答案: {numerical_answer}")
        return True, numerical_answer
    else:
        print("未能从回答中提取到数值答案")
        return False, response

async def async_stepwise_answer_solver(user_question, model_name="o3-mini"):
    """
    Use a step-by-step approach to solve optimization problems directly.
    """
    # Step 1: Build mathematical model
    step1_messages = [
        {"role": "system", "content": (
            "你是一个运筹优化专家。请根据用户提供的运筹优化问题构建数学模型。"
            "请明确定义决策变量、目标函数和约束条件。"
            "用数学表达式清晰地表达模型。"
        )},
        {"role": "user", "content": user_question}
    ]
    
    math_model = await async_query_llm(step1_messages, model_name)
    print("【数学模型】:\n", math_model)
    
    # Step 2: Solve the model
    step2_messages = [
        {"role": "system", "content": (
            "你是一个运筹优化专家。基于已建立的数学模型，请直接计算并求解优化问题。"
            "请给出最优解的目标函数值。"
            "在回答的最后，请以JSON格式给出最终答案: ```json\n{\"answer\": [数值]}\n```，其中[数值]是最优目标函数值。"
        )},
        {"role": "user", "content": f"问题: {user_question}\n\n已建立的数学模型:\n{math_model}\n\n请基于此模型直接求解，给出最优目标函数值。"}
    ]
    
    solution = await async_query_llm(step2_messages, model_name)
    print("【求解过程】:\n", solution)
    
    # Extract numerical answer from response
    numerical_answer = extract_numerical_answer(solution)
    
    if numerical_answer is not None:
        print(f"提取到的数值答案: {numerical_answer}")
        return True, numerical_answer
    else:
        print("未能从回答中提取到数值答案")
        return False, solution

async def process_single_case(i, d, args):
    """
    Process a single test case
    """
    start_time = time.time()
    
    
    user_question, answer = d['question'], d['answer']
    
    
    if args.stepwise:
        is_solve_success, llm_result = await async_stepwise_answer_solver(user_question, args.model)
    else:
        is_solve_success, llm_result = await async_direct_answer_solver(user_question, args.model)
        
    if is_solve_success:
        print(f"成功获得答案: {llm_result}")
    else:
        print("未能获得有效数值答案。")
    print('------------------')
    
    pass_flag, correct_flag = eval_model_result(is_solve_success, llm_result, answer)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"=============== num {i} ==================")
    print(user_question)
    print('-------------')
    print(f'solve: {is_solve_success}, llm: {llm_result}, ground truth: {answer}')
    print(f'[Final] run pass: {pass_flag}, solve correct: {correct_flag}')
    print(f'[Time] Processing time: {elapsed_time:.2f} seconds')
    print(' ')
    
    return llm_result, pass_flag, correct_flag, i, elapsed_time

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run optimization problem solving with direct LLM answers (async version)')
    parser.add_argument('--stepwise', action='store_true', 
                        help='Use step-by-step approach (build model first, then solve). If not specified, directly ask for answer.')
    parser.add_argument('--model', type=str, default='o3-mini',
                        help='Model name to use for LLM queries. Use "claude-..." for Claude models.')
    parser.add_argument('--data_path', type=str, default='data/datasets/dataset_combined_result.json',
                        help='Path to the dataset JSON file')
    return parser.parse_args()

async def main():
    # Load dataset from JSON file
    args = parse_args()
    
    with open(args.data_path, 'r') as f:
        dataset = json.load(f)
    
    # Create tasks for all test cases
    tasks = []
    for i, d in dataset.items():
    # if int(i) < 3:
        task = process_single_case(i, d, args)
        tasks.append(task)
    
    # Run all tasks concurrently and gather results
    results = await asyncio.gather(*tasks)
    
    # Process results
    pass_count = sum(1 for _, pass_flag, _, _, _ in results if pass_flag)
    correct_count = sum(1 for _, _, correct_flag, _, _ in results if correct_flag)
    error_datas = [i for _, pass_flag, correct_flag, i, _ in results if not pass_flag or not correct_flag]
    total_time = sum(elapsed_time for _, _, _, _, elapsed_time in results)
    avg_time = total_time / len(results) if len(results) > 0 else 0
    
    print(f'[Total {len(dataset)}] run pass: {pass_count}, solve correct: {correct_count}')
    print(f'[Total fails {len(error_datas)}] error datas: {error_datas}')
    print(f'[Timing] Total processing time: {total_time:.2f} seconds')
    print(f'[Timing] Average time per case: {avg_time:.2f} seconds')

if __name__ == "__main__":
    # Running as script
    asyncio.run(main()) 