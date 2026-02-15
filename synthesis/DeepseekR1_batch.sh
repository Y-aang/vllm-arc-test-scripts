#!/bin/bash

# Set experiment parameters
model_name="DeepSeek-R1-Distill-Qwen-1.5B"      # Model name (modifiable)
dataset_name="squad"      # Dataset name (modifiable)
sample_strategy="DistShift"  # Sampling strategy (modifiable)
cache_strategy="dbl"       # Cache strategy (modifiable: "lru" or "arc" or "dbl")

# Output directory path (auto-organized)
output_dir="./test/${model_name}/${dataset_name}/${sample_strategy}/${cache_strategy}"

# Delete if directory already exists
# if [ -d "$output_dir" ]; then
#     rm -rf "$output_dir"
#     echo "Deleted existing directory: $output_dir"
# fi

# Create output directory
mkdir -p "$output_dir"

# Custom batch size sequence
batch_sizes=(1 2 4 8 16 32 64 128)

# Iterate over batch sizes
for batch_size in "${batch_sizes[@]}"
do
    # Generate output file path
    output_file="${output_dir}/${batch_size}.txt"
    
    # Call Python script and save output to specified file
    python test_script_batch.py --batch_size $batch_size > "$output_file" 2>&1
    
    # Print current execution status
    echo "Completed experiment with batch_size=${batch_size}, saved to ${output_file}"
done

echo "All experiments completed."

