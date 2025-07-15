#!/usr/bin/env python3
"""
Script to convert Default.json and Default-en.json to IndustryOR.json format.

This script combines data from both files to create a new format where:
- "id": comes from "index" field
- "en_question": comes from "question" field in Default-en.json  
- "cn_question": comes from "question" field in Default.json
- "en_answer": comes from "answer" field (same in both files)
- "difficulty": always set to "Medium"
"""

import json
import os

def convert_default_to_industry_format(cn_file, en_file, output_file):
    """
    Convert Default.json and Default-en.json to IndustryOR.json format.
    
    Args:
        cn_file (str): Path to Default.json (Chinese questions)
        en_file (str): Path to Default-en.json (English questions)
        output_file (str): Path to output file
    """
    
    # Read the Chinese version (Default.json)
    with open(cn_file, 'r', encoding='utf-8') as f:
        cn_data = json.load(f)
    
    # Read the English version (Default-en.json)  
    with open(en_file, 'r', encoding='utf-8') as f:
        en_data = json.load(f)
    
    # Convert to IndustryOR.json format
    converted_data = []
    
    # Iterate through all entries (assuming both files have same keys)
    for key in cn_data.keys():
        if key in en_data:
            cn_entry = cn_data[key]
            en_entry = en_data[key]
            
            # Create new entry in IndustryOR format
            new_entry = {
                "en_question": en_entry["question"],
                "cn_question": cn_entry["question"], 
                "en_answer": str(en_entry["answer"]),
                "difficulty": "Medium",
                "id": cn_entry["index"]
            }
            
            converted_data.append(new_entry)
        else:
            print(f"Warning: Key {key} not found in English data")
    
    # Sort by id to maintain order
    converted_data.sort(key=lambda x: x["id"])
    
    # Write output file in the same format as IndustryOR.json (one JSON object per line)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in converted_data:
            json.dump(entry, f, ensure_ascii=False, separators=(',', ':'))
            f.write('\n')
    
    print(f"Successfully converted {len(converted_data)} entries.")
    print(f"Output written to: {output_file}")

def main():
    """Main function to run the conversion."""
    
    # Define file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cn_file = os.path.join(script_dir, "datasets", "Default.json")
    en_file = os.path.join(script_dir, "datasets", "Default-en.json") 
    output_file = os.path.join(script_dir, "datasets", "Default_converted.json")
    
    # Check if input files exist
    if not os.path.exists(cn_file):
        print(f"Error: Chinese file not found: {cn_file}")
        return
        
    if not os.path.exists(en_file):
        print(f"Error: English file not found: {en_file}")
        return
    
    # Run conversion
    try:
        convert_default_to_industry_format(cn_file, en_file, output_file)
    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    main() 