import pandas as pd
import matplotlib.pyplot as plt
import os

# Read data file
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, 'result', 'statistics.txt')
output_dir = os.path.join(script_dir, 'result')

# Read CSV data
df = pd.read_csv(data_file)

# Create figure
plt.figure(figsize=(12, 8))

# Define colors
colors = {
    'ARC': {'Miss': '#1f77b4', 'Hit': '#ff7f0e'},
    'LRU': {'Miss': '#2ca02c', 'Hit': '#d62728'},
    'DBL': {'Miss': '#9467bd', 'Hit': '#8c564b'}  # Add colors for DBL
}

# Default color list (for undefined types)
default_miss_colors = ['#17becf', '#bcbd22', '#e377c2', '#7f7f7f']
default_hit_colors = ['#c5b0d5', '#c49c94', '#f7b6d3', '#c7c7c7']

def get_color(evictor_type, color_type):
    """Get color; use default color if undefined."""
    if evictor_type in colors and color_type in colors[evictor_type]:
        return colors[evictor_type][color_type]
    else:
        # Use default color (selected based on hash of evictor_type)
        default_colors = default_miss_colors if color_type == 'Miss' else default_hit_colors
        idx = hash(evictor_type) % len(default_colors)
        return default_colors[idx]

# Plot line chart for each EvictorType
for evictor_type in df['EvictorType'].unique():
    data = df[df['EvictorType'] == evictor_type].sort_values('PromptLength')
    
    # Plot Miss_Mean
    plt.plot(data['PromptLength'], data['Miss_Mean'], 
             marker='o', linewidth=2, label=f'{evictor_type} Miss_Mean',
             color=get_color(evictor_type, 'Miss'))
    
    # Plot Hit_Mean
    plt.plot(data['PromptLength'], data['Hit_Mean'], 
             marker='s', linewidth=2, label=f'{evictor_type} Hit_Mean',
             color=get_color(evictor_type, 'Hit'))

# Set figure properties
plt.xlabel('PromptLength', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.title('Micro Benchmark: Miss_TTFT and Hit_TTFT by EvictorType', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save first image (containing Miss and Hit)
output_file = os.path.join(output_dir, 'micro_bench_graph.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'Chart saved to: {output_file}')

plt.close()

# Create second figure (showing Hit only)
plt.figure(figsize=(12, 8))

# Plot Hit line chart for each EvictorType
for evictor_type in df['EvictorType'].unique():
    data = df[df['EvictorType'] == evictor_type].sort_values('PromptLength')
    
    # Plot Hit_Mean only
    plt.plot(data['PromptLength'], data['Hit_Mean'], 
             marker='s', linewidth=2, label=f'{evictor_type} Hit_Mean',
             color=get_color(evictor_type, 'Hit'))

# Set figure properties
plt.xlabel('PromptLength', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.title('Micro Benchmark: Hit_TTFT by EvictorType', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save second image (showing Hit only)
output_file_hit = os.path.join(output_dir, 'micro_bench_graph_hit.png')
plt.savefig(output_file_hit, dpi=300, bbox_inches='tight')
print(f'Chart saved to: {output_file_hit}')

plt.close()

