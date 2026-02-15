#!/bin/bash

# Set script directory
SCRIPT_DIR="/home/shenyang/tests/verify"
RESULT_FILE="/home/shenyang/tests/verify/result/microbench.txt"
PYTHON_SCRIPT="${SCRIPT_DIR}/strategy_speed.py"

# Ensure result directory exists
mkdir -p "$(dirname "$RESULT_FILE")"

# Write header
echo "EvictorType,PromptLength,Run,Miss,Hit" > "$RESULT_FILE"

# Define arrays
evictor_types=(DBL)
prompt_lengths=(32 64 128 256 512 1024 2048 4096)
runs=(1 2 3 4 5 6 7 8)

# Outer loop: Evictor Type (LRU, ARC)
for evictor_type in "${evictor_types[@]}"; do
    echo "=========================================="
    echo "Switching to Evictor Type: $evictor_type"
    echo "=========================================="
    
    export VLLM_CUSTOMIZED_EVICTOR_TYPE="$evictor_type"
    
    # Second loop: Prompt Length (256, 512, 1024)
    for prompt_length in "${prompt_lengths[@]}"; do
        echo "----------------------------------------"
        echo "Prompt Length: $prompt_length"
        echo "----------------------------------------"
        
        # Third loop: repeat experiment 8 times
        for run in "${runs[@]}"; do
            echo "Running experiment $run/8 (Evictor: $evictor_type, Prompt Length: $prompt_length)"
            
            # Call Python script
            python "$PYTHON_SCRIPT" --prompt_length "$prompt_length" --block_size 16
            
            # Read the last line from the result file (just written result)
            if [ -f "$RESULT_FILE" ]; then
                # Get total line count
                total_lines=$(wc -l < "$RESULT_FILE")
                if [ "$total_lines" -gt 1 ]; then
                    # Read the last line (strip newline)
                    last_line=$(tail -n 1 "$RESULT_FILE" | tr -d '\n')
                    # If the last line is in numeric format (contains commas and is not a header), replace with metadata line
                    if [[ "$last_line" =~ ^[0-9]+\.[0-9]+,[0-9]+\.[0-9]+$ ]]; then
                        # Use sed to replace the last line
                        sed -i "\$s/.*/$evictor_type,$prompt_length,$run,$last_line/" "$RESULT_FILE"
                    fi
                fi
            fi
        done
        
        echo "Completed all experiments for Prompt Length $prompt_length"
        echo ""
    done
    
    echo "Completed all experiments for Evictor Type $evictor_type"
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: $RESULT_FILE"
echo "=========================================="

