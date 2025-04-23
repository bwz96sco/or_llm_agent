import openai
import anthropic
from dotenv import load_dotenv
import os
import re
import subprocess
import sys
import tempfile
import copy
import json
import argparse
from itertools import zip_longest
from utils import (
    is_number_string,
    convert_to_number,
    extract_best_objective,
    extract_and_execute_python_code,
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
openai_client = openai.OpenAI(
    api_key=openai_api_data['api_key'],
    base_url=openai_api_data['base_url'] if openai_api_data['base_url'] else None
)

anthropic_client = anthropic.Anthropic(
    api_key=anthropic_api_data['api_key']
)

def query_llm(messages, model_name="o3-mini", temperature=0.2):
    """
    调用 LLM 获取响应结果。
    
    Args:
        messages (list): 对话上下文列表。
        model_name (str): LLM模型名称，默认为"o3-mini"。
        temperature (float): 控制输出的随机性，默认为 0.2。

    Returns:
        str: LLM 生成的响应内容。
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

        response = anthropic_client.messages.create(
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
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content

def generate_or_code_solver(messages_bak, model_name, max_attempts):
    messages = copy.deepcopy(messages_bak)

    gurobi_code = query_llm(messages, model_name)
    print("【Python Gurobi 代码】:\n", gurobi_code)

    # 4. 代码执行 & 修复
    text = f"{gurobi_code}"
    attempt = 0
    while attempt < max_attempts:
        success, error_msg = extract_and_execute_python_code(text)
        if success:
            messages_bak.append({"role": "assistant", "content": gurobi_code})
            return True, error_msg, messages_bak

        print(f"\n第 {attempt + 1} 次尝试失败，请求 LLM 修复代码...\n")

        # 构建修复请求
        messages.append({"role": "assistant", "content": gurobi_code})
        messages.append({"role": "user", "content": f"代码执行出现错误，错误信息如下:\n{error_msg}\n请修复代码并重新提供完整的可执行代码。"})

        # 获取修复后的代码
        gurobi_code = query_llm(messages, model_name)
        text = f"{gurobi_code}"

        print("\n获取到修复后的代码，准备重新执行...\n")
        attempt += 1
    # not add gurobi code
    messages_bak.append({"role": "assistant", "content": gurobi_code})
    print(f"达到最大尝试次数 ({max_attempts})，未能成功执行代码。")
    return False, None, messages_bak

def or_llm_agent(user_question, model_name="o3-mini", max_attempts=3):
    """
    向 LLM 请求 Gurobi 代码解决方案并执行，如果失败则尝试修复。

    Args:
        user_question (str): 用户的问题描述。
        model_name (str): 使用的 LLM 模型名称，默认为"gpt-4"。
        max_attempts (int): 最大尝试次数，默认为3。

    Returns:
        tuple: (success: bool, best_objective: float or None, final_code: str)
    """
    # 初始化对话记录
    messages = [
        {"role": "system", "content": (
            "你是一个运筹优化专家。请根据用户提供的运筹优化问题构建数学模型，以数学（线性规划）模型对原问题进行有效建模。"
            "尽量关注获得一个正确的数学模型表达式，无需太关注解释。"
            "该模型后续用作指导生成gurobi代码，这一步主要用作生成有效的线性规模表达式。"
        )},
        {"role": "user", "content": user_question}
    ]

    # 1. 生成数学模型
    math_model = query_llm(messages, model_name)
    print("【数学模型】:\n", math_model)

    # # 2. 校验数学模型
    # messages.append({"role": "assistant", "content": math_model})
    # messages.append({"role": "user", "content": (
    #     "请基于上面的数学模型是否符合问题描述，如果存在错误，则进行修正；如果不存在错误则检查是否能进行优化。"
    #     "无论何种情况，最终请重新输出该数学模型。"
    # )})

    # validate_math_model = query_llm(messages, model_name)
    # print("【校验后的数学模型】:\n", validate_math_model)
    
    validate_math_model = math_model
    messages.append({"role": "assistant", "content": validate_math_model})
    
    # ------------------------------
    messages.append({"role": "user", "content": (
        "请基于以上的数学模型，写出完整、可靠的 Python 代码，使用 Gurobi 求解该运筹优化问题。"
        "代码中请包含必要的模型构建、变量定义、约束添加、目标函数设定以及求解和结果输出。"
        "以 ```python\n{code}\n``` 形式输出，无需输出代码解释。"
    )})
    # copy msg; solve; add the laset gurobi code 
    is_solve_success, result, messages = generate_or_code_solver(messages, model_name,max_attempts)
    print(f'Stage result: {is_solve_success}, {result}')
    if is_solve_success:
        if not is_number_string(result):
            print('!![No available solution warning]!!')
            # no solution 
            messages.append({"role": "user", "content": (
                "现有模型运行结果为*无可行解*，请认真仔细地检查数学模型和gurobi代码，是否存在错误，以致于造成无可行解"
                "检查完成后，最终请重新输出gurobi python代码"
                "以 ```python\n{code}\n``` 形式输出，无需输出代码解释。"
            )})
            is_solve_success, result, messages = generate_or_code_solver(messages, model_name, max_attempts=1)
    else:
        print('!![Max attempt debug error warning]!!')
        messages.append({"role": "user", "content": (
                "现在模型代码多次调试仍然报错，请认真仔细地检查数学模型是否存在错误"
                "检查后最终请重新构建gurobi python代码"
                "以 ```python\n{code}\n``` 形式输出，无需输出代码解释。"
            )})
        is_solve_success, result, messages = generate_or_code_solver(messages, model_name, max_attempts=2)
    
    return is_solve_success, result

def gpt_code_agent_simple(user_question, model_name="o3-mini", max_attempts=3):
    """
    向 LLM 请求 Gurobi 代码解决方案并执行，如果失败则尝试修复。

    Args:
        user_question (str): 用户的问题描述。
        model_name (str): 使用的 LLM 模型名称，默认为"gpt-4"。
        max_attempts (int): 最大尝试次数，默认为3。

    Returns:
        tuple: (success: bool, best_objective: float or None, final_code: str)
    """
    # 初始化对话记录
    messages = [
        {"role": "system", "content": (
            "你是一个运筹优化专家。请根据用户提供的运筹优化问题构建数学模型，并写出完整、可靠的 Python 代码，使用 Gurobi 求解该运筹优化问题。"
            "代码中请包含必要的模型构建、变量定义、约束添加、目标函数设定以及求解和结果输出。"
                "以 ```python\n{code}\n``` 形式输出，无需输出代码解释。"
        )},
        {"role": "user", "content": user_question}
    ]

    # copy msg; solve; add the laset gurobi code
    gurobi_code = query_llm(messages, model_name)
    print("【Python Gurobi 代码】:\n", gurobi_code)
    text = f"{gurobi_code}"
    is_solve_success, result = extract_and_execute_python_code(text)
    
    print(f'Stage result: {is_solve_success}, {result}')
    
    return is_solve_success, result

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run optimization problem solving with LLMs')
    parser.add_argument('--agent', action='store_true', 
                        help='Use the agent. If not specified, directly use the model to solve the problem')
    parser.add_argument('--model', type=str, default='o3-mini',
                        help='Model name to use for LLM queries. Use "claude-..." for Claude models.')
    parser.add_argument('--data_path', type=str, default='data/datasets/dataset_combined_result.json',
                        help='Path to the dataset JSON file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    with open(args.data_path, 'r') as f:
        dataset = json.load(f)
    #print(dataset['0'])

    model_name = args.model

    pass_count = 0
    correct_count = 0
    error_datas = []
    for i, d in dataset.items():
        print(f"=============== num {i} ==================")
        user_question, answer = d['question'], d['answer']
        print(user_question)
        print('-------------')
        
        if args.agent:
            is_solve_success, llm_result = or_llm_agent(user_question, model_name)
        else:
            is_solve_success, llm_result = gpt_code_agent_simple(user_question, model_name)
            
        if is_solve_success:
            print(f"成功执行代码，最优解值: {llm_result}")
        else:
            print("执行代码失败。")
        print('------------------')
        pass_flag, correct_flag = eval_model_result(is_solve_success, llm_result, answer)

        pass_count += 1 if pass_flag else 0
        correct_count += 1 if correct_flag else 0

        if not pass_flag or not correct_flag:
            error_datas.append(i)

        print(f'solve: {is_solve_success}, llm: {llm_result}, ground truth: {answer}')
        print(f'[Final] run pass: {pass_flag}, solve correct: {correct_flag}')
        print(' ')
            
    print(f'[Total {len(dataset)}] run pass: {pass_count}, solve correct: {correct_count}')
    print(f'[Total fails {len(error_datas)}] error datas: {error_datas}')