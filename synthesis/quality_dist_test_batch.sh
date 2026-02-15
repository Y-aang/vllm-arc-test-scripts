#!/bin/bash

# ========== Experiment Configuration ==========
model_name="Qwen14B_WikiQA"   # Model name (modifiable)
dataset_name="Quality"                           # Dataset name (modifiable)
sample_strategy="Distshift"                        # Sampling strategy (modifiable)

script_name="test_script_batch.py"            # Python script to call
param_name="cache_size"                         # Test parameter name
block_size=16                                 # Number of tokens per block, modifiable

# ========== Test Configuration ==========
cache_size=18750                                    # Fixed cache size (blocks)
# cache_size=12500                                    # Fixed cache size (blocks)
actual_cache_size=$((cache_size * block_size))     # Actual cache size (tokens)
batch_sizes=(8 1)                            # Batch size list (in order)
evictor_types=(LRU ARC DBL)                        # Evictor types list

# ========== Run Experiments in Loop ==========
# Outer loop: batch size
for batch_size in "${batch_sizes[@]}"
do
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
        
        output_file="${output_dir}/${actual_cache_size}_${batch_size}.txt"
        
        echo "🚀 Running with cache_size=${actual_cache_size} tokens (block_size=${block_size}), batch_size=${batch_size}, evictor_type=${evictor_type}, cache_strategy=${cache_strategy}..."
        
        # Call Python script and redirect output
        python "${script_name}" --${param_name} ${actual_cache_size} --batch_size ${batch_size} > "${output_file}" 2>&1
        
        echo "✅ Completed experiment with cache_size=${actual_cache_size}, batch_size=${batch_size}, evictor_type=${evictor_type}, results saved to ${output_file}"
    done
done

echo "🎯 All batch size and evictor type experiments completed."
