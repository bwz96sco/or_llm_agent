import openai
from dotenv import load_dotenv
import os
import re
import subprocess
import sys
import tempfile
import copy
import json
import shutil
import wcwidth
import json
from rich.console import Console
from rich.markdown import Markdown
import io 
from contextlib import redirect_stdout
import time
from utils import (
    is_number_string,
    convert_to_number,
    extract_best_objective,
    extract_and_execute_python_code,
    eval_model_result
)
# Load environment variables from .env file
load_dotenv()

# api_data = dict(
# api_key = 'sk-sbxihlsgrzsjfknusvrxiokzdwxofzbhjdyfznqgqifguclu', #os.getenv("OPENAI_API_KEY")
# base_url = 'https://api.siliconflow.cn/v1' #os.getenv("OPENAI_API_BASE")
# )    

# api_data = dict(
# api_key = os.getenv("OPENAI_API_KEY"),
# base_url = os.getenv("OPENAI_API_BASE")
# )   

api_data = dict(
    api_key = os.getenv("OPENAI_API_KEY")
)   

# Initialize OpenAI client
client = openai.OpenAI(
    api_key=api_data['api_key'],
)

def get_display_width(text):
    """
    Calculate the display width of a string, accounting for wide characters like Chinese.
    Uses the wcwidth module for accurate width calculation.
    
    Args:
        text (str): The text to calculate the width for.
        
    Returns:
        int: The display width of the text.
    """
    return wcwidth.wcswidth(text)


def print_header(text="", add_newline_before=True, add_newline_after=True, 
                border_char="=", side_char="||"):
    """
    Print a header with customizable text in the middle, adjusted to the console window width.
    Properly handles wide characters like Chinese.
    
    Args:
        text (str): The text to display in the middle of the header.
        add_newline_before (bool): Whether to add a newline before the header.
        add_newline_after (bool): Whether to add a newline after the header.
        border_char (str): Character to use for the top and bottom borders.
        side_char (str): Character to use for the side borders.
    """
    # Add a newline before the header if requested
    if add_newline_before:
        print()
    
    # Get terminal width
    # try:
    terminal_width = shutil.get_terminal_size().columns
    # except Exception:
    #     # Fallback width if terminal size cannot be determined
    #     terminal_width = 80
    
    # Ensure minimum width
    terminal_width = max(terminal_width, 40)
    
    # Calculate side character padding
    side_char_len = len(side_char)
    
    # Print the top border
    print(border_char * terminal_width)
    
    # Print the empty line
    print(side_char + " " * (terminal_width - 2 * side_char_len) + side_char)
    
    # Print the middle line with text
    text_display_width = get_display_width(text)
    available_space = terminal_width - 2 * side_char_len
    
    if text_display_width <= available_space:
        left_padding = (available_space - text_display_width) // 2
        right_padding = available_space - text_display_width - left_padding
        # print(terminal_width, text_display_width, available_space, left_padding, right_padding)
        print(side_char + " " * left_padding + text + " " * right_padding + side_char)
    else:
        # If text is too long, we need to truncate it
        # This is more complex with wide characters, so we'll do it character by character
        truncated_text = ""
        truncated_width = 0
        for char in text:
            char_width = get_display_width(char)
            if truncated_width + char_width + 3 > available_space:  # +3 for the "..."
                break
            truncated_text += char
            truncated_width += char_width
        
        truncated_text += "..."
        right_padding = available_space - get_display_width(truncated_text)
        print(side_char + truncated_text + " " * right_padding + side_char)
    
    # Print the empty line
    print(side_char + " " * (terminal_width - 2 * side_char_len) + side_char)
    
    # Print the bottom border
    print(border_char * terminal_width)
    
    # Add a newline after the header if requested
    if add_newline_after:
        print()

def query_llm(messages, model_name="o3-mini", temperature=0.2):
    """
    调用 LLM 获取响应结果，使用流式输出方式。
    
    Args:
        messages (list): 对话上下文列表。
        model_name (str): LLM模型名称，默认为"gpt-4"。
        temperature (float): 控制输出的随机性，默认为 0.2。

    Returns:
        str: LLM 生成的响应内容。
    """
    # 使用stream=True启用流式输出
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        stream=True
    )
    
    # 用于累积完整响应
    full_response = ""
    
    # 用于控制打印格式
    print("LLM Output: ", end="", flush=True)
    
    # 逐块处理流式响应
    for chunk in response:
        # 首先检查choices列表是否非空
        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
            # 然后检查是否有delta和content
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
                    full_response += content
    
    # 输出完成后换行
    print()
    
    return full_response

def generate_or_code_solver(messages_bak, model_name, max_attempts):
    messages = copy.deepcopy(messages_bak)
    
    print_header("LLM生成Python Gurobi 代码")

    gurobi_code = query_llm(messages, model_name)

    print_header("自动执行python代码")
    # 4. 代码执行 & 修复
    text = f"{gurobi_code}"
    attempt = 0
    while attempt < max_attempts:
        buffer2 = io.StringIO()
        with redirect_stdout(buffer2):
            success, error_msg = extract_and_execute_python_code(text)
        captured_output2 = buffer2.getvalue()
        for c in captured_output2:
            print(c, end="", flush=True)
            time.sleep(0.005)

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
    print_header("LLM推理构建线性规划模型")
    math_model = query_llm(messages, model_name)
    # print("【数学模型】:\n", math_model)

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



if __name__ == "__main__":
    # Import the load_dataset function from the async script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from or_llm_eval_async_resilient import load_dataset
    
    dataset = load_dataset('data/datasets/IndustryOR.json')
    # print(dataset['0'])
    console = Console()

    model_name = 'o3-mini'
    # model_name = ''

    # model_name = 'Pro/deepseek-ai/DeepSeek-R1' 
    # model_name = 'deepseek-reasoner'

    pass_count = 0
    correct_count = 0
    for i, d in dataset.items():
        #print(i)
        # if int(i) in [0]:
        print_header("运筹优化问题")
        user_question, answer = d['question'], d['answer']
        # print(user_question)
        buffer2 = io.StringIO()
        with redirect_stdout(buffer2):
            md = Markdown(user_question)
            console.print(md)
            print('-------------')

        captured_output2 = buffer2.getvalue()
        for c in captured_output2:
            print(c, end="", flush=True)
            time.sleep(0.005)
        is_solve_success, llm_result = or_llm_agent(user_question, model_name)
        # is_solve_success, llm_result = gpt_code_agent_simple(user_question, model_name)
        if is_solve_success:
            print(f"成功执行代码，最优解值: {llm_result}")
        else:
            print("执行代码失败。")