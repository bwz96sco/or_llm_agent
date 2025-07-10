#!/bin/bash

# Shell script to run OR-LLM evaluation on multiple datasets with logging
# This script runs the async evaluation with o3 model on two different datasets

echo "Starting OR-LLM evaluation batch process..."
echo "Timestamp: $(date)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# First evaluation: dataset_md_result.json
echo "Running evaluation on data/datasets/ORLM/IndustryOR.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json..."
echo "Command: python or_llm_eval_async.py --model o3 --data_path data/datasets/ORLM/IndustryOR.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
echo "Log file: logs/eval_IndustryOR_result_$(date +%Y%m%d_%H%M%S).log"

python or_llm_eval_async.py --model o3 --data_path data/datasets/ORLM/IndustryOR.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json > logs/eval_IndustryOR_result_$(date +%Y%m%d_%H%M%S).log 2>&1

if [ $? -eq 0 ]; then
    echo "✓ First evaluation completed successfully"
else
    echo "✗ First evaluation failed with exit code $?"
fi

echo ""

# Second evaluation: removed.json
echo "Running evaluation on data/datasets/ORLM/MAMO.ComplexLP.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json..."
echo "Command: python or_llm_eval_async.py --model o3 --data_path data/datasets/ORLM/MAMO.ComplexLP.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
echo "Log file: logs/eval_ComplexLP_$(date +%Y%m%d_%H%M%S).log"

python or_llm_eval_async.py --model o3 --data_path data/datasets/ORLM/MAMO.ComplexLP.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json > logs/eval_ComplexLP_$(date +%Y%m%d_%H%M%S).log 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Second evaluation completed successfully"
else
    echo "✗ Second evaluation failed with exit code $?"
fi

# Third evaluation: removed.json
echo "Running evaluation on data/datasets/ORLM/MAMO.EasyLP.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json..."
echo "Command: python or_llm_eval_async.py --model o3 --data_path data/datasets/ORLM/MAMO.EasyLP.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
echo "Log file: logs/eval_EasyLP_$(date +%Y%m%d_%H%M%S).log"

python or_llm_eval_async.py --model o3 --data_path data/datasets/ORLM/MAMO.EasyLP.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json > logs/eval_EasyLP_$(date +%Y%m%d_%H%M%S).log 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Second evaluation completed successfully"
else
    echo "✗ Second evaluation failed with exit code $?"
fi

# Fourth evaluation: removed.json
echo "Running evaluation on data/datasets/ORLM/NL4OPT.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json..."
echo "Command: python or_llm_eval_async.py --model o3 --data_path data/datasets/ORLM/NL4OPT.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json"
echo "Log file: logs/eval_NL4OPT_$(date +%Y%m%d_%H%M%S).log"

python or_llm_eval_async.py --model o3 --data_path data/datasets/ORLM/NL4OPT.q2mc_en.ORLM-LLaMA-3-8B/executed_converted.json > logs/eval_NL4OPT_$(date +%Y%m%d_%H%M%S).log 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Second evaluation completed successfully"
else
    echo "✗ Second evaluation failed with exit code $?"
fi

echo ""
echo "=========================================="
echo "Batch evaluation process completed!"
echo "Timestamp: $(date)"
echo "Check the logs/ directory for detailed output files" 