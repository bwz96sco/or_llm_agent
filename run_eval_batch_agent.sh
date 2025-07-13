#!/bin/bash

# Shell script to run OR-LLM evaluation on multiple datasets with configurable parameters

echo "Starting OR-LLM evaluation batch process..."
echo "Timestamp: $(date)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Configuration array: [agent_mode, model, data_path]
# If agent_mode is empty, --agent flag will not be added
# If agent_mode is non-empty (e.g., "agent"), --agent flag will be added

# configurations=(
#     #GPT-4o
#     "agent,gpt-4o,data/datasets/dataset_combined_result.json"
#     "agent,gpt-4o,data/datasets/ORLM/IndustryOR.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
#     "agent,gpt-4o,data/datasets/ORLM/MAMO.ComplexLP.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
#     "agent,gpt-4o,data/datasets/ORLM/NL4OPT.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
#     "agent,gpt-4o,data/datasets/ORLM/MAMO.EasyLP.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"

#     #gemini-2.0-flash
#     "agent,gemini-2.0-flash,data/datasets/dataset_combined_result.json"
#     "agent,gemini-2.0-flash,data/datasets/ORLM/IndustryOR.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
#     "agent,gemini-2.0-flash,data/datasets/ORLM/MAMO.ComplexLP.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
#     "agent,gemini-2.0-flash,data/datasets/ORLM/NL4OPT.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
#     "agent,gemini-2.0-flash,data/datasets/ORLM/MAMO.EasyLP.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"

#     #deepseek/deepseek-r1-0528:free-V3
#     "agent,deepseek/deepseek-r1-0528:free-V3,data/datasets/dataset_combined_result.json"
#     "agent,deepseek/deepseek-r1-0528:free-V3,data/datasets/ORLM/IndustryOR.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
#     "agent,deepseek/deepseek-r1-0528:free-V3,data/datasets/ORLM/MAMO.ComplexLP.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
#     "agent,deepseek/deepseek-r1-0528:free-V3,data/datasets/ORLM/NL4OPT.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
#     "agent,deepseek/deepseek-r1-0528:free-V3,data/datasets/ORLM/MAMO.EasyLP.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"

#     #deepseek/deepseek-r1-0528:free-R1
#     "agent,deepseek/deepseek-r1-0528:free-R1,data/datasets/dataset_combined_result.json"
#     "agent,deepseek/deepseek-r1-0528:free-R1,data/datasets/ORLM/IndustryOR.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
#     "agent,deepseek/deepseek-r1-0528:free-R1,data/datasets/ORLM/MAMO.ComplexLP.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
#     "agent,deepseek/deepseek-r1-0528:free-R1,data/datasets/ORLM/NL4OPT.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
#     "agent,deepseek/deepseek-r1-0528:free-R1,data/datasets/ORLM/MAMO.EasyLP.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
# )
configurations=(
    #deepseek/deepseek-r1-0528:free-R1
    "agent,deepseek/deepseek-r1-0528:free-R1,data/datasets/ORLM/MAMO.ComplexLP.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
    "agent,deepseek/deepseek-r1-0528:free-R1,data/datasets/ORLM/MAMO.EasyLP.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
)

# Counter for evaluation numbering
eval_count=1
total_evals=${#configurations[@]}

# Loop through each configuration
for config in "${configurations[@]}"; do
    # Parse the configuration (split by comma)
    IFS=',' read -r agent_mode model data_path <<< "$config"
    
    echo "Running evaluation $eval_count/$total_evals..."
    echo "Configuration: agent_mode='$agent_mode', model='$model', data_path='$data_path'"
    
    # Build the command
    cmd="python or_llm_eval_async.py"
    
    # Add --agent flag if agent_mode is not empty
    if [ -n "$agent_mode" ]; then
        cmd="$cmd --agent"
    fi
    
    # Add model and data_path parameters
    cmd="$cmd --model $model --data_path $data_path"
    
    # Extract dataset name from data_path for log naming
    dataset_name=$(basename $(dirname "$data_path"))
    
    # Sanitize model name for filesystem (replace / and : with -)
    sanitized_model=$(echo "$model" | sed 's/[\/:]/-/g')
    
    # Generate log filename with agent mode, model, and dataset name
    if [ -n "$agent_mode" ]; then
        log_file="logs/eval_${dataset_name}_${agent_mode}_${sanitized_model}_$(date +%Y%m%d_%H%M%S).log"
    else
        log_file="logs/eval_${dataset_name}_${sanitized_model}_$(date +%Y%m%d_%H%M%S).log"
    fi
    
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