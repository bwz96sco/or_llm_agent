#!/usr/bin/env python3
"""
Script to regenerate sequential index numbers in a JSON file.
Usage: python regenerate_index.py <json_file_path>
"""

import json
import sys
import argparse
from pathlib import Path


def regenerate_index(json_file_path):
    """
    Regenerate sequential index numbers in a JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file
    """
    # Load the JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{json_file_path}': {e}")
        return False
    
    # Convert to list of items and sort by current index to maintain order
    items = []
    for key, value in data.items():
        if isinstance(value, dict) and 'index' in value:
            items.append(value)
        else:
            print(f"Warning: Item with key '{key}' doesn't have an 'index' field")
    
    # Sort by current index to maintain the original order
    items.sort(key=lambda x: x.get('index', float('inf')))
    
    # Create new data with regenerated indices
    new_data = {}
    for i, item in enumerate(items):
        item['index'] = i  # Update the index field
        new_data[str(i)] = item  # Use string keys as in original
    
    # Create backup of original file
    backup_path = f"{json_file_path}.backup"
    try:
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Backup created: {backup_path}")
    except Exception as e:
        print(f"Warning: Could not create backup: {e}")
    
    # Write the regenerated data back to the file
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)
        
        print(f"Successfully regenerated indices in '{json_file_path}'")
        print(f"Total items: {len(new_data)}")
        print(f"Index range: 0 to {len(new_data) - 1}")
        return True
        
    except Exception as e:
        print(f"Error writing to file '{json_file_path}': {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate sequential index numbers in a JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python regenerate_index.py data.json
  python regenerate_index.py /path/to/dataset.json
        """
    )
    parser.add_argument(
        'json_file', 
        help='Path to the JSON file to process'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Show what would be changed without modifying the file'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.json_file).exists():
        print(f"Error: File '{args.json_file}' does not exist.")
        sys.exit(1)
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
        # Load and analyze the file
        try:
            with open(args.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            items = []
            for key, value in data.items():
                if isinstance(value, dict) and 'index' in value:
                    items.append((key, value['index']))
            
            items.sort(key=lambda x: x[1])
            
            print(f"Current structure:")
            print(f"Total items: {len(items)}")
            print(f"Key -> Index mapping:")
            for key, index in items[:10]:  # Show first 10
                print(f"  '{key}' -> {index}")
            if len(items) > 10:
                print(f"  ... and {len(items) - 10} more items")
            
            print(f"\nAfter regeneration, indices would be: 0 to {len(items) - 1}")
            
        except Exception as e:
            print(f"Error analyzing file: {e}")
            sys.exit(1)
    else:
        success = regenerate_index(args.json_file)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main() 