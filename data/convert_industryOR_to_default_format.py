#!/usr/bin/env python3
"""
Script to convert executed.jsonl to the same format as dataset_combined_result.json
Converts "en_question" to "question" and "en_answer" to "answer", and generates sequential indices.
"""

import json
import os

def convert_executed_to_dataset_format(input_file, output_file):
    """
    Convert executed.jsonl format to dataset_combined_result.json format
    
    Args:
        input_file (str): Path to the input executed.jsonl file
        output_file (str): Path to the output JSON file
    """
    
    converted_data = {}
    index = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    # Parse the JSON line
                    data = json.loads(line)
                    
                    # Convert the format
                    converted_entry = {
                        "index": index,
                        "question": data.get("en_question", ""),
                        "answer": data.get("en_answer", "")
                    }
                    
                    # Add to the converted data dictionary
                    converted_data[str(index)] = converted_entry
                    index += 1
                    
                    # Print progress every 100 entries
                    if index % 100 == 0:
                        print(f"Processed {index} entries...")
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    print(f"Problematic line: {line[:100]}...")
                    continue
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error reading input file: {e}")
        return False
    
    # Write the converted data to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=4)
        
        print(f"Conversion completed successfully!")
        print(f"Total entries processed: {index}")
        print(f"Output saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error writing output file: {e}")
        return False

def main():
    # Define file paths
    # input_file = "data/datasets/ORLM/IndustryOR.q2mc_en.ORLM-LLaMA-3-8B/executed.jsonl"
    # output_file = "data/datasets/ORLM/IndustryOR.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
    # input_file = "data/datasets/ORLM/MAMO.ComplexLP.q2mc_en.ORLM-LLaMA-3-8B/executed.jsonl"
    # output_file = "data/datasets/ORLM/MAMO.ComplexLP.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
    # input_file = "data/datasets/ORLM/MAMO.EasyLP.q2mc_en.ORLM-LLaMA-3-8B/executed.jsonl"
    # output_file = "data/datasets/ORLM/MAMO.EasyLP.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
    input_file = "data/datasets/ORLM/NL4OPT.q2mc_en.ORLM-LLaMA-3-8B/executed.jsonl"
    output_file = "data/datasets/ORLM/NL4OPT.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return
    
    print(f"Converting {input_file} to {output_file}...")
    print("=" * 50)
    
    # Perform the conversion
    success = convert_executed_to_dataset_format(input_file, output_file)
    
    if success:
        print("=" * 50)
        print("Conversion completed successfully!")
        
        # Show a sample of the converted data
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                converted_data = json.load(f)
                
            print(f"\nSample of converted data (first entry):")
            if "0" in converted_data:
                sample_entry = converted_data["0"]
                print(f"Index: {sample_entry['index']}")
                print(f"Question (first 100 chars): {sample_entry['question'][:100]}...")
                print(f"Answer: {sample_entry['answer']}")
            
        except Exception as e:
            print(f"Error reading converted file for sample: {e}")
    else:
        print("Conversion failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 