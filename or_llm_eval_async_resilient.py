import openai
import anthropic
from dotenv import load_dotenv
import os
import re
import subprocess
import sys
import tempfile

import json
import asyncio
import argparse
from itertools import zip_longest
from utils import (
    is_number_string,
    convert_to_number,
    extract_best_objective,
    eval_model_result
)

# Prompt constants
MATH_MODEL_SYSTEM_PROMPT = (
    "你是一个运筹优化专家。请根据用户提供的运筹优化问题构建数学模型，以数学（线性规划）模型对原问题进行有效建模。"
    "尽量关注获得一个正确的数学模型表达式，无需太关注解释。"
    "该模型后续用作指导生成gurobi代码，这一步主要用作生成有效的线性规模表达式。"
)

CODE_GENERATION_SYSTEM_PROMPT = (
    "你是一个运筹优化专家。请根据用户提供的运筹优化问题构建数学模型，并写出完整、可靠的 Python 代码，使用 Gurobi 求解该运筹优化问题。"
    "代码中请包含必要的模型构建、变量定义、约束添加、目标函数设定以及求解和结果输出。"
    "以 ```python\n{code}\n``` 形式输出，无需输出代码解释。"
)

REQUEST_GUROBI_CODE_WITH_MATH_PROMPT = (
    "请基于以上的数学模型，写出完整、可靠的 Python 代码，使用 Gurobi 求解该运筹优化问题。"
    "代码中请包含必要的模型构建、变量定义、约束添加、目标函数设定以及求解和结果输出。"
    "以 ```python\n{code}\n``` 形式输出，无需输出代码解释。"
)

ERROR_FIX_PROMPT_TEMPLATE = (
    "代码执行出现错误，错误信息如下:\n{error_msg}\n请修复代码并重新提供完整的可执行代码。"
)

INFEASIBLE_SOLUTION_PROMPT = (
    "现有模型运行结果为*无可行解*，请认真仔细地检查数学模型和gurobi代码，是否存在错误，以致于造成无可行解"
    "检查完成后，最终请重新输出gurobi python代码"
    "以 ```python\n{code}\n``` 形式输出，无需输出代码解释。"
)

MAX_ATTEMPT_ERROR_PROMPT = (
    "现在模型代码多次调试仍然报错，请认真仔细地检查数学模型是否存在错误"
    "检查后最终请重新构建gurobi python代码"
    "以 ```python\n{code}\n``` 形式输出，无需输出代码解释。"
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

async def async_query_llm(messages, model_name="o3-mini", temperature=0.2, max_attempts=3):
    """
    Async version of query_llm that supports both OpenAI and Claude models with retry functionality.
    Returns (success, result) tuple instead of raising exceptions after max attempts.
    """
    import time
    
    for attempt in range(max_attempts):
        try:
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
                return True, response.content[0].text
            else:
                # Use OpenAI API
                response = await openai_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature
                )
                return True, response.choices[0].message.content
                
        except (openai.APIConnectionError, anthropic.APIConnectionError) as e:
            print(f"[Connection Error] Attempt {attempt + 1}/{max_attempts} failed for model {model_name}")
            print(f"[Connection Error] Error details: {str(e)}")
            print(f"[Connection Error] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if attempt < max_attempts - 1:
                # Exponential backoff: 2^attempt seconds (2, 4, 8, ...)
                wait_time = 60 * (attempt + 1)
                print(f"[Connection Error] Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                print(f"[Connection Error] Max attempts ({max_attempts}) reached. Continuing with failure.")
                return False, f"Connection error after {max_attempts} attempts: {str(e)}"
                
        except Exception as e:
            # For other types of errors, don't retry
            print(f"[API Error] Non-connection error occurred with model {model_name}: {str(e)}")
            print(f"[API Error] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return False, f"API error: {str(e)}"
    
    return False, "Unknown error occurred"

async def async_extract_and_execute_python_code(text_content):
    """
    Async version of extract_and_execute_python_code with 60-second timeout
    """
    python_code_blocks = re.findall(r'```python\s*([\s\S]*?)```', text_content)

    if not python_code_blocks:
        print("未找到Python代码块。")
        return False, "No Python code blocks found"

    for code_block in python_code_blocks:
        code_block = code_block.strip()
        if not code_block:
            print("找到一个空的Python代码块，已跳过。")
            continue

        print("找到Python代码块，开始执行...")
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
                tmp_file.write(code_block)
                temp_file_path = tmp_file.name

            # Add timeout to prevent infinite loops or long-running code
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    temp_file_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Wait for process completion with 60-second timeout
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)
                
            except asyncio.TimeoutError:
                print("Python 代码执行超时 (60秒)，可能存在无限循环或长时间运行的代码")
                # Kill the process if it's still running
                try:
                    proc.kill()
                    await proc.wait()
                except:
                    pass
                return False, "Code execution timeout (60 seconds) - possible infinite loop"
            
            if proc.returncode == 0:
                print("Python 代码执行成功，输出:\n")
                stdout_str = stdout.decode()
                print(stdout_str)
                
                best_obj = extract_best_objective(stdout_str)
                if best_obj is not None:
                    print(f"\n最优解值 (Best objective): {best_obj}")
                else:
                    print("\n未找到最优解值")
                return True, str(best_obj)
            else:
                print(f"Python 代码执行出错，错误信息:\n")
                print(stderr.decode())
                return False, stderr.decode()

        except Exception as e:
            print(f"执行Python代码块时发生错误: {e}")
            return False, str(e)
        finally:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        print("-" * 30)

    return False, "No valid code blocks executed"

async def async_generate_math_model(user_question, model_name="o3-mini"):
    """
    Generate mathematical model from user question
    """
    messages = [
        {"role": "system", "content": MATH_MODEL_SYSTEM_PROMPT},
        {"role": "user", "content": user_question}
    ]

    success, math_model = await async_query_llm(messages, model_name)
    if not success:
        print(f"数学模型生成失败: {math_model}")
        return False, f"MATH_MODEL_FAILED: {math_model}"
    
    #print("【数学模型】:\n", math_model)
    return True, math_model

async def async_generate_and_run_gurobi_code(user_question, model_name="o3-mini", math_model=None, enable_debug=False, max_attempts=3):
    """
    Generate Gurobi code either from user question directly or with math model
    """
    if math_model:
        # Build conversation with math model context
        messages = [
            {"role": "system", "content": MATH_MODEL_SYSTEM_PROMPT},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": math_model},
            {"role": "user", "content": REQUEST_GUROBI_CODE_WITH_MATH_PROMPT}
        ]
    else:
        # Direct code generation from user question
        messages = [
            {"role": "system", "content": CODE_GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_question}
        ]

    # Simple single-attempt version
    actual_attemps = 1
    if enable_debug:
        # Use the debug-enabled version with multiple attempts
        actual_attemps = max_attempts

    is_solve_success, result, executed_content = await async_generate_or_code_solver(messages, model_name, actual_attemps)
    messages.append({"role": "assistant", "content": executed_content})
    return is_solve_success, result, messages, executed_content

async def async_generate_or_code_solver(messages, model_name, max_attempts):
    """
    Async version that handles LLM failures gracefully with multiple attempts to fix code errors
    """
    success, gurobi_code = await async_query_llm(messages, model_name)
    if not success:
        print(f"LLM生成Gurobi代码失败: {gurobi_code}")
        return False, f"CODE_GEN_ERROR: {gurobi_code}", ""
    
    #print("【Python Gurobi 代码】:\n", gurobi_code)

    executed_content = f"{gurobi_code}"
    attempt = 0
    while attempt < max_attempts:
        #error_msg is the error message when code execution failed, otherwise it is the best objective value
        success_exec, error_msg = await async_extract_and_execute_python_code(executed_content)
        if success_exec:
            return True, error_msg, executed_content

        print(f"\n第 {attempt + 1} 次尝试失败，请求 LLM 修复代码...\n")

        messages.append({"role": "assistant", "content": gurobi_code})
        messages.append({"role": "user", "content": ERROR_FIX_PROMPT_TEMPLATE.format(error_msg=error_msg)})

        success, gurobi_code = await async_query_llm(messages, model_name)
        if not success:
            print(f"LLM生成Gurobi代码失败: {gurobi_code}")
            return False, f"CODE_GEN_ERROR: {gurobi_code}", ""
        
        executed_content = f"{gurobi_code}"

        print("\n获取到修复后的代码，准备重新执行...\n")
        attempt += 1

    print(f"达到最大尝试次数 ({max_attempts})，未能成功执行代码。")
    return False, error_msg, executed_content

async def async_or_llm_agent(user_question, model_name="o3-mini", use_math_model=False, enable_debug=False, max_attempts=3):
    """
    Main agent function that orchestrates math model generation and code generation
    """
    math_model = None
    
    # Step 1: Generate math model if requested
    if use_math_model:
        success, math_model = await async_generate_math_model(user_question, model_name)
        if not success:
            return False, math_model, None, ""
    
    # Step 2: Generate and execute Gurobi code
    is_solve_success, result, messages, executed_content = await async_generate_and_run_gurobi_code(
        user_question, model_name, math_model, enable_debug, max_attempts
    )
    
    # Step 3: Handle unsuccessful results with additional debugging (only if debug enabled and math model was used)
    if use_math_model and enable_debug:
        if is_solve_success:
            if not is_number_string(result):
                print('!![No available solution warning]!!')
                # Try to fix infeasible solution
                messages.append({"role": "user", "content": INFEASIBLE_SOLUTION_PROMPT})
                is_solve_success, result, executed_content = await async_generate_or_code_solver(messages, model_name, max_attempts=1)
        else:
            if not result.startswith("CODE_GEN_ERROR"):
                print('!![Max attempt debug error warning]!!')
                messages.append({"role": "user", "content": MAX_ATTEMPT_ERROR_PROMPT})
                is_solve_success, result, executed_content = await async_generate_or_code_solver(messages, model_name, max_attempts=2)
    
    return is_solve_success, result, math_model, executed_content


async def process_single_case(i, d, args):
    """
    Process a single test case with failure tracking
    """
    user_question, answer = d['question'], d['answer']
    failure_reason = None
    
    try:
        is_solve_success, llm_result, math_model, executed_content = await async_or_llm_agent(
            user_question, 
            args.model, 
            use_math_model=args.math,
            enable_debug=args.debug
        )
            
        if is_solve_success:
            print(f"成功执行代码，最优解值: {llm_result}")
        else:
            print("执行代码失败。")
            if isinstance(llm_result, str) and llm_result.startswith("CODE_GEN_ERROR"):
                failure_reason = "CODE_GEN_ERROR"
            elif isinstance(llm_result, str) and llm_result.startswith("MATH_MODEL_FAILED"):
                failure_reason = "MATH_MODEL_FAILED"
            else:
                failure_reason = "CODE_EXECUTION_FAILED"
        
        print('------------------')
        
        pass_flag, correct_flag = eval_model_result(is_solve_success, llm_result, answer)
        
        print(f"=============== num {i} ==================")
        print(user_question)
        print('math model: ------------------------------------------------------------------------------------------------')
        print(f'{math_model}')
        print('executed content: ------------------------------------------------------------------------------------------------')
        print(f'{executed_content}')
        print('------------------------------------------------------------------------------------------------')
        print(f'solve: {is_solve_success}, llm: {llm_result}, ground truth: {answer}')
        print(f'[Final] run pass: {pass_flag}, solve correct: {correct_flag}')
        print("=================================================================================================")
        
        return llm_result, pass_flag, correct_flag, i, failure_reason
    
    except Exception as e:
        print(f"Task {i} failed with unexpected error: {str(e)}")
        failure_reason = f"UNEXPECTED_ERROR: {str(e)}"
        return None, False, False, i, failure_reason

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run optimization problem solving with LLMs (resilient async version)')
    parser.add_argument('--math', action='store_true', 
                        help='Generate mathematical model first, then use it to guide code generation')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debugging mode with multiple attempts to fix code errors')
    parser.add_argument('--model', type=str, default='o3-mini',
                        help='Model name to use for LLM queries. Use "claude-..." for Claude models.')
    parser.add_argument('--data_path', type=str, default='data/datasets/IndustryOR.json',
                        help='Path to the dataset JSON file (supports both JSONL and regular JSON formats)')
    return parser.parse_args()

def load_dataset(data_path):
    """
    Load dataset from either JSONL format (IndustryOR.json, BWOR.json) or regular JSON format
    """
    dataset = {}
    
    with open(data_path, 'r', encoding='utf-8') as f:
        # Try to detect format by reading first line
        first_line = f.readline().strip()
        f.seek(0)  # Reset file pointer
        
        if first_line.startswith('{"en_question"') or first_line.startswith('{"cn_question"'):
            # JSONL format (IndustryOR.json, BWOR.json)
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        # Convert to expected format
                        dataset_item = {
                            'question': item.get('en_question', item.get('cn_question', '')),
                            'answer': item.get('en_answer', item.get('cn_answer', '')),
                            'difficulty': item.get('difficulty', 'Unknown'),
                            'id': item.get('id', line_num - 1)
                        }
                        # Use id as string key
                        dataset[str(dataset_item['id'])] = dataset_item
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line {line_num}: {line}")
                        continue
        else:
            # Regular JSON format (legacy)
            dataset = json.load(f)
    
    return dataset

async def main():
    # Load dataset from JSON file
    args = parse_args()
    
    dataset = load_dataset(args.data_path)
    
    # Convert dataset items to a list for easier chunking
    dataset_items = list(dataset.items())
    
    # Process dataset in batches of 50
    batch_size = 50
    all_results = []
    total_batches = (len(dataset_items) + batch_size - 1) // batch_size  # Ceiling division
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(dataset_items))
        batch_items = dataset_items[start_idx:end_idx]
        
        print(f"\n{'='*50}")
        print(f"Processing batch {batch_num + 1}/{total_batches} (cases {start_idx + 1}-{end_idx})")
        print(f"{'='*50}\n")
        
        # Create tasks for current batch
        tasks = []
        for i, d in batch_items:
            task = process_single_case(i, d, args)
            tasks.append(task)
        
        # Run current batch concurrently and gather results
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in batch results
        processed_batch_results = []
        for idx, result in enumerate(batch_results):
            if isinstance(result, Exception):
                task_id = batch_items[idx][0]  # Get the task ID
                print(f"Exception in task {task_id}: {str(result)}")
                processed_batch_results.append((None, False, False, task_id, f"EXCEPTION: {str(result)}"))
            else:
                processed_batch_results.append(result)
        
        all_results.extend(processed_batch_results)
        
        # Print batch summary
        batch_pass_count = sum(1 for _, pass_flag, _, _, _ in processed_batch_results if pass_flag)
        batch_correct_count = sum(1 for _, _, correct_flag, _, _ in processed_batch_results if correct_flag)
        print(f"\nBatch {batch_num + 1} Summary:")
        print(f"  Processed: {len(processed_batch_results)} cases")
        print(f"  Run pass: {batch_pass_count}")
        print(f"  Solve correct: {batch_correct_count}")
        print(f"  Completed: {end_idx}/{len(dataset_items)} total cases")
    
    # Process final results
    pass_count = sum(1 for _, pass_flag, _, _, _ in all_results if pass_flag)
    correct_count = sum(1 for _, _, correct_flag, _, _ in all_results if correct_flag)
    error_datas = [i for _, pass_flag, correct_flag, i, _ in all_results if not pass_flag or not correct_flag]
    
    # Track failure types
    failure_summary = {}
    failed_tasks = []
    for llm_result, pass_flag, correct_flag, task_id, failure_reason in all_results:
        if not pass_flag or not correct_flag:
            failed_tasks.append({
                'task_id': task_id,
                'pass_flag': pass_flag,
                'correct_flag': correct_flag,
                'failure_reason': failure_reason or 'UNKNOWN',
                'llm_result': llm_result
            })
            
            failure_type = failure_reason or 'UNKNOWN'
            if failure_type not in failure_summary:
                failure_summary[failure_type] = []
            failure_summary[failure_type].append(task_id)
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f'[Total {len(dataset)}] run pass: {pass_count}, solve correct: {correct_count}')
    print(f'[Total fails {len(error_datas)}] error datas: {error_datas}')
    
    print(f"\n{'='*50}")
    print("FAILURE ANALYSIS")
    print(f"{'='*50}")
    
    if failure_summary:
        print(f"Total failed tasks: {len(failed_tasks)}")
        print("\nFailure breakdown by type:")
        for failure_type, task_ids in failure_summary.items():
            print(f"  {failure_type}: {len(task_ids)} tasks")
            print(f"    Task IDs: {task_ids}")
        
        print(f"\nDetailed failure list:")
        for i, failure in enumerate(failed_tasks[:20]):  # Show first 20 failures
            print(f"  {i+1}. Task {failure['task_id']}: {failure['failure_reason']}")
            if failure['llm_result'] and len(str(failure['llm_result'])) < 100:
                print(f"     Result: {failure['llm_result']}")
        
        if len(failed_tasks) > 20:
            print(f"     ... and {len(failed_tasks) - 20} more failures")
    else:
        print("No failures detected!")

if __name__ == "__main__":
    # Running as script
    asyncio.run(main()) 