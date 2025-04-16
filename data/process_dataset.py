#!/usr/bin/env python3
import os
import json
import glob
import argparse

def process_dataset(dataset_path, output_file, is_numbered=True):
    """
    Process a dataset containing problem directories with description.txt and sample.json files.
    
    Args:
        dataset_path (str): Path to the dataset directory
        output_file (str): Path to the output JSON file
        is_numbered (bool): Whether the directories have numeric indices (e.g., prob_10) or names
    """
    # Get all problem directories
    if is_numbered:
        prob_dirs = glob.glob(os.path.join(dataset_path, "prob_*"))
        # Sort directories by problem number for numbered directories
        prob_dirs.sort(key=lambda x: int(x.split("_")[-1]))
    else:
        prob_dirs = glob.glob(os.path.join(dataset_path, "*"))
        # Filter out any non-directory items
        prob_dirs = [d for d in prob_dirs if os.path.isdir(d)]
        # Sort directories alphabetically for named directories
        prob_dirs.sort()
    
    # Dictionary to store all the processed problems
    combined_data = {}
    
    # Process each problem directory
    for i, prob_dir in enumerate(prob_dirs):
        # Extract problem identifier (directory name or number)
        prob_id = os.path.basename(prob_dir)
        
        # Paths to the description and sample files
        description_path = os.path.join(prob_dir, "description.txt")
        sample_path = os.path.join(prob_dir, "sample.json")
        
        # Check if both files exist
        if not (os.path.exists(description_path) and os.path.exists(sample_path)):
            print(f"Warning: Missing files in {prob_dir}. Skipping.")
            continue
        
        # Read the question from description.txt
        with open(description_path, 'r', encoding='utf-8') as f:
            question = f.read().strip()
        
        # Read the answer and input from sample.json
        try:
            with open(sample_path, 'r', encoding='utf-8') as f:
                sample_data = json.load(f)
                # Assuming the output is the answer we want
                if isinstance(sample_data, list) and len(sample_data) > 0:
                    if 'output' in sample_data[0]:
                        answer = sample_data[0]['output']
                        # If answer is a list with one item, extract that item
                        if isinstance(answer, list) and len(answer) == 1:
                            answer = answer[0]
                    else:
                        print(f"Warning: No 'output' field in {prob_dir}/sample.json. Skipping.")
                        continue
                    
                    # Add input data to the question if available
                    if 'input' in sample_data[0]:
                        input_data = sample_data[0]['input']
                        # Format input data as a string and append to question
                        input_str = json.dumps(input_data, indent=2)
                        question += f"\n\nInput:\n{input_str}"
                else:
                    print(f"Warning: Unexpected sample.json format in {prob_dir}. Skipping.")
                    continue
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in {prob_dir}/sample.json. Skipping.")
            continue
        
        # Add to combined data
        combined_data[str(i)] = {
            "index": i,
            "question": question,
            "answer": answer
        }
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write the combined data to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)
    
    print(f"Processing complete. Output written to {output_file}")
    print(f"Processed {len(combined_data)} problems out of {len(prob_dirs)} directories")

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Process a dataset of problems into a combined JSON file.')
    parser.add_argument('dataset_path', help='Path to the dataset directory')
    parser.add_argument('output_file', help='Path to the output JSON file')
    parser.add_argument('--named', action='store_true', help='Use if directories have names instead of numeric indices')
    
    args = parser.parse_args()
    
    # Process the dataset
    process_dataset(args.dataset_path, args.output_file, not args.named) 