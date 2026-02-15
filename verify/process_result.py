import pandas as pd
import numpy as np
from pathlib import Path

# Read data file
data_file = Path(__file__).parent / 'result' / 'microbench_300.txt'
df = pd.read_csv(data_file)

# Filter out empty rows
df = df.dropna()

# Group by EvictorType and PromptLength
grouped = df.groupby(['EvictorType', 'PromptLength'])

# Calculate statistics
results = []
for (evictor_type, prompt_length), group in grouped:
    miss_values = group['Miss'].values
    hit_values = group['Hit'].values
    
    miss_mean = np.mean(miss_values)
    miss_std = np.std(miss_values, ddof=1)  # Use sample standard deviation (n-1)
    hit_mean = np.mean(hit_values)
    hit_std = np.std(hit_values, ddof=1)
    
    results.append({
        'EvictorType': evictor_type,
        'PromptLength': prompt_length,
        'Miss_Mean': miss_mean,
        'Miss_Std': miss_std,
        'Hit_Mean': hit_mean,
        'Hit_Std': hit_std,
        'Num_Runs': len(group)
    })

# Create result DataFrame
result_df = pd.DataFrame(results)

# Print results
print("=" * 80)
print("Statistics: Mean and Std Dev of Miss and Hit for each EvictorType and PromptLength combination")
print("=" * 80)
print()
print(result_df.to_string(index=False))
print()

# Optionally save to file
output_file = Path(__file__).parent / 'result' / 'statistics.txt'
result_df.to_csv(output_file, index=False, float_format='%.6f')
print(f"Results saved to: {output_file}")

