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
import asyncio
import argparse
from itertools import zip_longest
from utils import (
    is_number_string,
    convert_to_number,
    extract_best_objective,
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

async def async_query_llm(messages, model_name="o3-mini", temperature=0.2, max_attempts=3):
    """
    Async version of query_llm that supports both OpenAI and Claude models with retry functionality
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
                return response.content[0].text
            else:
                # Use OpenAI API
                response = await openai_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature
                )
                return response.choices[0].message.content
                
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
                print(f"[Connection Error] Max attempts ({max_attempts}) reached. Giving up.")
                raise e
                
        except Exception as e:
            # For other types of errors, don't retry
            print(f"[API Error] Non-connection error occurred with model {model_name}: {str(e)}")
            print(f"[API Error] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            raise e

async def async_extract_and_execute_python_code(text_content):
    """
    Async version of extract_and_execute_python_code with 60-second timeout
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
                print("Python code execution timeout (60 seconds), possible infinite loop or long-running code")
                # Kill the process if it's still running
                try:
                    proc.kill()
                    await proc.wait()
                except:
                    pass
                return False, "Code execution timeout (60 seconds) - possible infinite loop"
            
            if proc.returncode == 0:
                print("Python code execution successful, output:\n")
                stdout_str = stdout.decode()
                print(stdout_str)
                
                best_obj = extract_best_objective(stdout_str)
                if best_obj is not None:
                    print(f"\nBest objective value: {best_obj}")
                else:
                    print("\nBest objective value not found")
                return True, str(best_obj)
            else:
                print(f"Python code execution error, error message:\n")
                print(stderr.decode())
                return False, stderr.decode()

        except Exception as e:
            print(f"Error occurred while executing Python code block: {e}")
            return False, str(e)
        finally:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        print("-" * 30)

    return False, "No valid code blocks executed"

async def async_gpt_code_agent_simple(user_question, model_name="o3-mini", max_attempts=3):
    """
    Async version of gpt_code_agent_simple
    """
    messages = [
        {"role": "system", "content": (
            "You are an operations research optimization expert. Please build a mathematical model based on the operations research optimization problem provided by the user, and write complete, reliable Python code using Gurobi to solve the operations research optimization problem."
            "The code should include necessary model construction, variable definition, constraint addition, objective function setting, solving, and result output."
            "Output in ```python\n{code}\n``` format, no need for code explanation."
        )},
        {"role": "user", "content": user_question}
    ]

    gurobi_code = await async_query_llm(messages, model_name)
    print("【Python Gurobi Code】:\n", gurobi_code)
    text = f"{gurobi_code}"
    is_solve_success, result = await async_extract_and_execute_python_code(text)
    
    print(f'Stage result: {is_solve_success}, {result}')
    
    return is_solve_success, result

async def async_generate_or_code_solver(messages_bak, model_name, max_attempts):
    messages = copy.deepcopy(messages_bak)

    gurobi_code = await async_query_llm(messages, model_name)
    print("【Python Gurobi Code】:\n", gurobi_code)

    text = f"{gurobi_code}"
    attempt = 0
    while attempt < max_attempts:
        success, error_msg = await async_extract_and_execute_python_code(text)
        if success:
            messages_bak.append({"role": "assistant", "content": gurobi_code})
            return True, error_msg, messages_bak

        print(f"\nAttempt {attempt + 1} failed, requesting LLM to fix the code...\n")

        messages.append({"role": "assistant", "content": gurobi_code})
        messages.append({"role": "user", "content": f"Code execution error occurred, error message as follows:\n{error_msg}\nPlease fix the code and provide the complete executable code again."})

        gurobi_code = await async_query_llm(messages, model_name)
        text = f"{gurobi_code}"

        print("\nReceived fixed code, preparing to re-execute...\n")
        attempt += 1

    messages_bak.append({"role": "assistant", "content": gurobi_code})
    print(f"Reached maximum attempts ({max_attempts}), failed to execute code successfully.")
    return False, None, messages_bak

async def async_or_llm_agent(user_question, model_name="o3-mini", max_attempts=3):
    """
    Async version of or_llm_agent function.
    """
    # Initialize conversation history
    messages = [
        {"role": "system", "content": (
            "You are an operations research optimization expert. Please build a mathematical model based on the operations research optimization problem provided by the user, and effectively model the original problem using a mathematical (linear programming) model."
            "Focus on obtaining a correct mathematical model expression, without too much emphasis on explanation."
            "This model will be used subsequently to guide the generation of gurobi code. This step is mainly used to generate effective linear programming expressions."
        )},
        {"role": "user", "content": user_question}
    ]

    # 1. Generate mathematical model
    math_model = await async_query_llm(messages, model_name)
    print("【Mathematical Model】:\n", math_model)
    
    validate_math_model = math_model
    messages.append({"role": "assistant", "content": validate_math_model})
    
    messages.append({"role": "user", "content": (
        "Please write complete, reliable Python code based on the above mathematical model, using Gurobi to solve this operations research optimization problem."
        "The code should include necessary model construction, variable definition, constraint addition, objective function setting, solving, and result output."
        "Output in ```python\n{code}\n``` format, no need for code explanation."
    )})

    # copy msg; solve; add the last gurobi code 
    is_solve_success, result, messages = await async_generate_or_code_solver(messages, model_name, max_attempts)
    print(f'Stage result: {is_solve_success}, {result}')
    
    if is_solve_success:
        if not is_number_string(result):
            print('!![No available solution warning]!!')
            messages.append({"role": "user", "content": (
                "The current model run result is *no feasible solution*, please carefully check the mathematical model and gurobi code for errors that may cause no feasible solution"
                "After checking, please output the gurobi python code again"
                "Output in ```python\n{code}\n``` format, no need for code explanation."
            )})
            is_solve_success, result, messages = await async_generate_or_code_solver(messages, model_name, max_attempts=1)
    else:
        print('!![Max attempt debug error warning]!!')
        messages.append({"role": "user", "content": (
                "The model code still has errors after multiple debugging attempts, please carefully check if there are errors in the mathematical model"
                "After checking, please rebuild the gurobi python code"
                "Output in ```python\n{code}\n``` format, no need for code explanation."
            )})
        is_solve_success, result, messages = await async_generate_or_code_solver(messages, model_name, max_attempts=2)
    
    return is_solve_success, result

async def process_single_case(i, d, args):
    """
    Process a single test case
    """
    print(f"=============== num {i} ==================")
    user_question, answer = d['question'], d['answer']
    print(user_question)
    print('-------------')
    
    if args.agent:
        is_solve_success, llm_result = await async_or_llm_agent(user_question, args.model)
    else:
        is_solve_success, llm_result = await async_gpt_code_agent_simple(user_question, args.model)
        
    if is_solve_success:
        print(f"Successfully executed code, optimal solution value: {llm_result}")
    else:
        print("Failed to execute code.")
    print('------------------')
    
    pass_flag, correct_flag = eval_model_result(is_solve_success, llm_result, answer)
    
    print(f'solve: {is_solve_success}, llm: {llm_result}, ground truth: {answer}')
    print(f'[Final] run pass: {pass_flag}, solve correct: {correct_flag}')
    print(' ')
    
    return llm_result, pass_flag, correct_flag, i

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run optimization problem solving with LLMs (async version)')
    parser.add_argument('--agent', action='store_true', 
                        help='Use the agent. If not specified, directly use the model to solve the problem')
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
    
    # Convert dataset items to a list for easier chunking
    dataset_items = list(dataset.items())
    
    # Process dataset in batches of 50
    batch_size = 20
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
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)
        
        # Print batch summary
        batch_pass_count = sum(1 for _, pass_flag, _, _ in batch_results if pass_flag)
        batch_correct_count = sum(1 for _, _, correct_flag, _ in batch_results if correct_flag)
        print(f"\nBatch {batch_num + 1} Summary:")
        print(f"  Processed: {len(batch_results)} cases")
        print(f"  Run pass: {batch_pass_count}")
        print(f"  Solve correct: {batch_correct_count}")
        print(f"  Completed: {end_idx}/{len(dataset_items)} total cases")
    
    # Process final results
    pass_count = sum(1 for _, pass_flag, _, _ in all_results if pass_flag)
    correct_count = sum(1 for _, _, correct_flag, _ in all_results if correct_flag)
    error_datas = [i for _, pass_flag, correct_flag, i in all_results if not pass_flag or not correct_flag]
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f'[Total {len(dataset)}] run pass: {pass_count}, solve correct: {correct_count}')
    print(f'[Total fails {len(error_datas)}] error datas: {error_datas}')

if __name__ == "__main__":
    # Running as script
    asyncio.run(main()) 