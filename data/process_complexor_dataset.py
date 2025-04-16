#!/usr/bin/env python3
import os
import json
import glob

def process_complexor_dataset():
    # Path to the ComplexOR dataset
    complexor_path = "data/datasets/ComplexOR"
    
    # Get all problem directories
    prob_dirs = glob.glob(os.path.join(complexor_path, "*"))
    # Filter out any non-directory items
    prob_dirs = [d for d in prob_dirs if os.path.isdir(d)]
    
    # Sort directories alphabetically
    prob_dirs.sort()
    
    # Dictionary to store all the processed problems
    combined_data = {}
    
    # Process each problem directory
    for i, prob_dir in enumerate(prob_dirs):
        # Extract problem name (directory name)
        prob_name = os.path.basename(prob_dir)
        
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
    
    # Write the combined data to a JSON file
    output_path = "data/datasets/complexor_combined_result.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)
    
    print(f"Processing complete. Output written to {output_path}")
    print(f"Processed {len(combined_data)} problems out of {len(prob_dirs)} directories")

if __name__ == "__main__":
    process_complexor_dataset() 