#!/bin/bash

# ========== Experiment Configuration ==========
model_name="DeepSeek-R1-Distill-Qwen-1.5B"   # Model name (modifiable)
dataset_name="wild"                           # Dataset name (modifiable)
sample_strategy="all_v1"                        # Sampling strategy (modifiable)

script_name="test_script_batch.py"            # Python script to call
param_name="cache_size"                         # Test parameter name
block_size=16                                 # Number of tokens per block, modifiable

# ========== Cache Size List for Testing ==========
# cache_sizes=(2000 5000 10000 15000 20000 38000)
# cache_sizes=(1875 3125 6250 12500 25000 37500)
# evictor_types=(LRU ARC DBL)
# runs=(1 2 3 4)   
cache_sizes=(18750)
evictor_types=(ARC)
runs=(1)                        # Number of repeated runs

# ========== Run Experiments in Loop ==========
# Outer loop: repeated runs
for run in "${runs[@]}"
do
    # Second loop: cache size
    for size in "${cache_sizes[@]}"
    do
        # Use block_size variable
        actual_size=$((size * block_size))
        
        # Inner loop: evictor types
        for evictor_type in "${evictor_types[@]}"
        do
            # Convert evictor_type to lowercase as cache_strategy
            cache_strategy=$(echo "$evictor_type" | tr '[:upper:]' '[:lower:]')
            
            # Build output directory based on cache_strategy
            output_dir="./test/${model_name}/${dataset_name}/${sample_strategy}/${cache_strategy}"
            
            # Create output directory
            mkdir -p "$output_dir"
            
            # Set environment variables
            export VLLM_CUSTOMIZED_EVICTOR_TYPE="$evictor_type"
            
            # Setup cache statistics JSON file path
            cache_stats_json="${output_dir}/${actual_size}_${run}_cache_stats.json"
            export CACHE_STATS_JSON_PATH="$cache_stats_json"
            
            output_file="${output_dir}/${actual_size}_${run}.txt"
            
            echo "🚀 Running (${run}/${#runs[@]}) with cache_size=${actual_size} tokens (block_size=${block_size}), evictor_type=${evictor_type}, cache_strategy=${cache_strategy}..."
            
            # Call Python script and redirect output
            python "${script_name}" --${param_name} ${actual_size} > "${output_file}" 2>&1
            
            echo "✅ Completed experiment (${run}/${#runs[@]}) with cache_size=${actual_size}, evictor_type=${evictor_type}, results saved to ${output_file}"
        done
    done
done

echo "🎯 All cache size and evictor type experiments completed."
