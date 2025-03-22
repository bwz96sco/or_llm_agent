import re
import subprocess
import sys
import tempfile
import os

def is_number_string(s):
    """
    判断字符串是否为数字字符串，包括整数和小数。

    Args:
    s: 要判断的字符串。

    Returns:
    如果字符串是数字字符串，则返回 True，否则返回 False。
    """
    pattern = r"^[-+]?\d+(\.\d+)?$"  # 匹配整数或小数的正则表达式
    return re.match(pattern, s) is not None

def convert_to_number(s):
    """
    将字符串转换为数字（整数或浮点数）。

    Args:
        s: 要转换的字符串。

    Returns:
        int or float: 如果字符串表示整数则返回int，如果表示小数则返回float。
        如果转换失败则返回None。
    """
    try:
        # 尝试转换为整数
        if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
            return int(s)
        # 尝试转换为浮点数
        num = float(s)
        return num
    except (ValueError, TypeError):
        return None

def extract_best_objective(output_text):
    """
    从Gurobi输出中提取Best objective或Optimal objective值。
    
    Args:
        output_text: Gurobi的输出文本
    
    Returns:
        float or None: 最优解值，如果未找到则返回None
    """
    # First check if model is infeasible
    if "Model is infeasible" in output_text:
        return None
    
    # Try to find Best objective
    match = re.search(r'Best objective\s+([\d.e+-]+)', output_text)
    if not match:
        # If not found, try to find Optimal objective
        match = re.search(r'Optimal objective\s+([\d.e+-]+)', output_text)
    
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    
    return None

def extract_and_execute_python_code(text_content):
    """
    从文本中提取Python代码块并执行。

    Args:
        text_content: 包含代码块的文本内容。

    Returns:
        bool: True if execution was successful, False otherwise
        str: Error message if execution failed, best objective if successful
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

            result = subprocess.run([sys.executable, temp_file_path], capture_output=True, text=True, check=False)

            if result.returncode == 0:
                print("Python 代码执行成功，输出:\n")
                print(result.stdout)
                
                best_obj = extract_best_objective(result.stdout)
                if best_obj is not None:
                    print(f"\n最优解值 (Best objective): {best_obj}")
                else:
                    print("\n未找到最优解值")
                return True, str(best_obj)
            else:
                print(f"Python 代码执行出错，错误信息:\n")
                print(result.stderr)
                return False, result.stderr

        except Exception as e:
            print(f"执行Python代码块时发生错误: {e}")
            return False, str(e)
        finally:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        print("-" * 30)

    return False, "No valid code blocks executed"

def eval_model_result(success, result, ground_truth, err_range=0.1):
    pass_flag = False
    correct_flag = False
    if success:
        pass_flag = True
        if is_number_string(str(result)) and ground_truth is not None:
            result_num = convert_to_number(str(result))
            ground_truth_num = convert_to_number(str(ground_truth))
            if abs(result_num - ground_truth_num) < err_range:
                correct_flag = True
        elif result == 'None': # no avai solution
            if ground_truth is None or ground_truth == 'None':
                correct_flag = True
    return pass_flag, correct_flag 