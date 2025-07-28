#!/bin/bash

# Shell script to run OR-LLM evaluation on multiple datasets with configurable parameters

echo "Starting OR-LLM evaluation batch process..."
echo "Timestamp: $(date)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Configuration array: [math_flag, debug_flag, model, data_path]
# math_flag: "math" to enable math model generation, empty string to disable
# debug_flag: "debug" to enable debugging mode, empty string to disable

configurations=(
    "math,debug,DeepSeek-R1,data/datasets/IndustryOR.json"
" , ,o3,data/datasets/IndustryOR.json"
)

# Counter for evaluation numbering
eval_count=1
total_evals=${#configurations[@]}

# Loop through each configuration
for config in "${configurations[@]}"; do
    # Parse the configuration (split by comma)
    IFS=',' read -r math_flag debug_flag model data_path <<< "$config"
    
    echo "Running evaluation $eval_count/$total_evals..."
    echo "Configuration: math_flag='$math_flag', debug_flag='$debug_flag', model='$model', data_path='$data_path'"
    
    # Build the command
    cmd="python or_llm_eval_async_resilient.py"
    
    # Add --math flag if math_flag is "math"
    if [ "$math_flag" = "math" ]; then
        cmd="$cmd --math"
    fi
    
    # Add --debug flag if debug_flag is "debug"
    if [ "$debug_flag" = "debug" ]; then
        cmd="$cmd --debug"
    fi
    
    # Add model and data_path parameters
    cmd="$cmd --model $model --data_path $data_path"
    
    # Extract dataset name from data_path for log naming
    dataset_name=$(basename "$data_path" .json)
    
    # Sanitize model name for filesystem (replace / and : with -)
    sanitized_model=$(echo "$model" | sed 's/[\/:]/-/g')
    
    # Generate log filename with math/debug flags, model, and dataset name
    mode_suffix=""
    if [ "$math_flag" = "math" ]; then
        mode_suffix="${mode_suffix}_math"
    fi
    if [ "$debug_flag" = "debug" ]; then
        mode_suffix="${mode_suffix}_debug"
    fi
    if [ -z "$mode_suffix" ]; then
        mode_suffix="_direct"
    fi
    
    log_file="logs/eval_${dataset_name}${mode_suffix}_${sanitized_model}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "Command: $cmd"
    echo "Log file: $log_file"
    
    # Execute the command
    eval "$cmd" > "$log_file" 2>&1
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✓ Evaluation $eval_count completed successfully"
    else
        echo "✗ Evaluation $eval_count failed with exit code $?"
    fi
    
    echo ""
    ((eval_count++))
done

echo "=========================================="
echo "Batch evaluation process completed!"
echo "Total evaluations run: $total_evals"
echo "Timestamp: $(date)"
echo "Check the logs/ directory for detailed output files" 