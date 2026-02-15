#!/bin/bash

# Set experiment parameters
model_name="Qwen2.5-1.5B-Instruct"      # Model name (modifiable)
dataset_name="squad"      # Dataset name (modifiable)
sample_strategy="hotspot"  # Sampling strategy (modifiable)
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

custom_sequence=(10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100)
# custom_sequence=(60 65 70 75 80 85 90 95 100)
# Loop CP_ratio from 5 to 50, with step size 5
for cp_ratio in "${custom_sequence[@]}"
do
    # Generate output file path
    output_file="${output_dir}/${cp_ratio}.txt"
    
    # Call Python script and save output to specified file
    python test_script.py --cp_ratio $cp_ratio > "$output_file" 2>&1
    
    # Print current execution status
    echo "Completed experiment with CP_ratio=${cp_ratio}, saved to ${output_file}"
done

echo "All experiments completed."
