#!/bin/bash

# ========== Experiment Configuration ==========
model_name="DeepSeek-R1-Distill-Qwen-1.5B"   # Model name (modifiable)
dataset_name="wild"                           # Dataset name (modifiable)
sample_strategy="all"                         # Sampling strategy (modifiable)
cache_strategy="lru"                          # Cache strategy ("lru" / "arc" / "dbl")

script_name="test_script_batch.py"            # Python script to call
param_name="cache_size"                       # Test parameter name
block_size=16                                 # Number of tokens per block, modifiable

# ========== Output Directory Structure ==========
output_dir="./test/${model_name}/${dataset_name}/${sample_strategy}/${cache_strategy}"

# Create output directory
mkdir -p "$output_dir"

# ========== Fixed Cache Size (unit: blocks) ==========
size=5000                                    # Fixed cache size (in blocks), modifiable

# Calculate actual token count
actual_size=$((size * block_size))
output_file_base="${output_dir}/${actual_size}"

# ========== Repeat execution ==========
for i in 1 2 3 4
do
    output_file="${output_file_base}_${i}.txt"
    echo "🚀 Running (${i}/4) with cache_size=${actual_size} tokens (block_size=${block_size})..."
    python "${script_name}" --${param_name} ${actual_size} > "${output_file}" 2>&1
    echo "✅ Completed run ${i}/8 with cache_size=${actual_size}, results saved to ${output_file}"
done

echo "🎯 All repeated runs completed. Results are in: ${output_dir}"