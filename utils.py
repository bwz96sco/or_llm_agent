import re
import subprocess
import sys
import tempfile
import os

def is_number_string(s):
    """
    Determine if a string is a numeric string, including integers and decimals.

    Args:
    s: The string to be checked.

    Returns:
    True if the string is a numeric string, otherwise False.
    """
    pattern = r"^[-+]?\d+(\.\d+)?$"  # Regular expression to match integers or decimals
    return re.match(pattern, s) is not None

def convert_to_number(s):
    """
    Convert a string to a number (integer or float).

    Args:
        s: The string to be converted.

    Returns:
        int or float: Returns int if the string represents an integer, float if it represents a decimal.
        Returns None if conversion fails.
    """
    try:
        # Try to convert to integer
        if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
            return int(s)
        # Try to convert to float
        num = float(s)
        return num
    except (ValueError, TypeError):
        return None

def extract_best_objective(output_text):
    """
    Extract Best objective or Optimal objective value from Gurobi output.
    
    Args:
        output_text: Gurobi output text
    
    Returns:
        float or None: Optimal solution value, returns None if not found
    """
    # First check if model is infeasible
    if "Model is infeasible" in output_text:
        return None
    
    # Try to find Best objective
    match = re.search(r'Best objective\s+([\d.e+-]+)', output_text)
    if not match:
        # If not found, try to find Optimal objective
        match = re.search(r'Optimal objective\s+([\d.e+-]+)', output_text)
    
    if not match:
        # If not found, try to find Optimal cost
        match = re.search(r'Optimal cost\s+([\d.e+-]+)', output_text)
    
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    
    return None

def extract_and_execute_python_code(text_content):
    """
    Extract Python code blocks from text and execute them.

    Args:
        text_content: Text content containing code blocks.

    Returns:
        bool: True if execution was successful, False otherwise
        str: Error message if execution failed, best objective if successful
    """
    python_code_blocks = re.findall(r'```python\s*([\s\S]*?)```', text_content)

    if not python_code_blocks:
        print("No Python code blocks found.")
        return False, "No Python code blocks found"

    for code_block in python_code_blocks:
        code_block = code_block.strip()
        if not code_block:
            print("Found an empty Python code block, skipped.")
            continue

        print("Found Python code block, starting execution...")
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
                tmp_file.write(code_block)
                temp_file_path = tmp_file.name

            result = subprocess.run([sys.executable, temp_file_path], capture_output=True, text=True, check=False)

            if result.returncode == 0:
                print("Python code executed successfully, output:\n")
                print(result.stdout)
                
                best_obj = extract_best_objective(result.stdout)
                if best_obj is not None:
                    print(f"\nOptimal solution value (Best objective): {best_obj}")
                else:
                    print("\nOptimal solution value not found")
                return True, str(best_obj)
            else:
                print(f"Python code execution error, error message:\n")
                print(result.stderr)
                return False, result.stderr

        except Exception as e:
            print(f"Error occurred while executing Python code block: {e}")
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
            # Check that both conversions were successful before comparing
            if result_num is not None and ground_truth_num is not None:
                if abs(result_num - ground_truth_num) < err_range:
                    correct_flag = True
        elif result == 'None': # no available solution
            if ground_truth is None or ground_truth == 'None':
                correct_flag = True
    return pass_flag, correct_flag 