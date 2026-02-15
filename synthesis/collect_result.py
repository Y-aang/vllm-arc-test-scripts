import os
import re
import pandas as pd

def calculate_hitrate_after_cold_start(lines):
    """
    Calculate hit rate after cold start
    
    Args:
        lines: All lines of the log file
        
    Returns:
        float or None: Hit rate after cold start, or None if it cannot be calculated
    """
    # Find the position of the [cold start] line
    cold_start_line_idx = None
    for i, line in enumerate(lines):
        if "[cold start]" in line:
            cold_start_line_idx = i
            break
    
    if cold_start_line_idx is None:
        return None
    
    # Last hit and access data before cold start
    hit1 = None
    access1 = None
    for i in range(cold_start_line_idx - 1, -1, -1):
        if "hit:" in lines[i] and "access:" in lines[i]:
            match_hit = re.search(r"hit:\s*([0-9]+)", lines[i])
            match_access = re.search(r"access:\s*([0-9]+)", lines[i])
            if match_hit and match_access:
                hit1 = int(match_hit.group(1))
                access1 = int(match_access.group(1))
                break
    
    # Last hit and access data in the entire file
    hit2 = None
    access2 = None
    for line in reversed(lines):
        if "hit:" in line and "access:" in line:
            match_hit = re.search(r"hit:\s*([0-9]+)", line)
            match_access = re.search(r"access:\s*([0-9]+)", line)
            if match_hit and match_access:
                hit2 = int(match_hit.group(1))
                access2 = int(match_access.group(1))
                break
    
    # Calculate hit rate after cold start
    if hit1 is not None and access1 is not None and hit2 is not None and access2 is not None:
        if access2 - access1 > 0:
            return (hit2 - hit1) / (access2 - access1)
    
    return None

# Set variables
model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
dataset_name = "Length"   # wild  Quality
sample_strategy = "Random" # all  Distshift
cache_strategies = ["arc", "dbl", "lru", "lru_l"]
# cache_strategies = ["lru", "arc"]
cache_strategies = ["lru_l"]

# Initialize result table, key is Cache_Size (e.g. 5, 10...)
results = {}

for cache_strategy in cache_strategies:
    dir_path = f"./test/{model_name}/{dataset_name}/{sample_strategy}/{cache_strategy}"
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"{dir_path} not found")

    for filename in os.listdir(dir_path):
        if not filename.endswith(".txt"):
            continue
        cache_size = int(filename[:-4])
        file_path = os.path.join(dir_path, filename)

        with open(file_path, "r") as f:
            lines = f.readlines()

        # Find the last occurrence of gpu_hit_rate and two types of latency
        hit_rate = None
        latency_total = None
        latency_after_cold_start = None
        hitrate_after_cold_start = calculate_hitrate_after_cold_start(lines)
        
        for line in reversed(lines):
            if "gpu_hit_rate" in line:
                match = re.search(r"gpu_hit_rate:\s*([0-9.]+)", line)
                if match:
                    hit_rate = float(match.group(1))
                    break

        # Find the last occurrence of "Average of first token latencies (after cold start):"
        for line in reversed(lines):
            if "Average of first token latencies (after cold start):" in line:
                match = re.search(r"Average of first token latencies \(after cold start\):\s*([0-9.]+)", line)
                if match:
                    latency_after_cold_start = float(match.group(1))
                    break

        # Find the last occurrence of "Average of first token latencies:" (excluding after cold start)
        for line in reversed(lines):
            if "Average of first token latencies:" in line and "(after cold start)" not in line:
                match = re.search(r"Average of first token latencies:\s*([0-9.]+)", line)
                if match:
                    latency_total = float(match.group(1))
                    break

        if cache_size not in results:
            results[cache_size] = {}

        results[cache_size][f"{cache_strategy.upper()}_hitrate"] = hit_rate
        results[cache_size][f"{cache_strategy.upper()}_time_total"] = latency_total
        results[cache_size][f"{cache_strategy.upper()}_hitrate_after_cold"] = hitrate_after_cold_start
        results[cache_size][f"{cache_strategy.upper()}_time_after_cold"] = latency_after_cold_start

# Build final DataFrame
df_rows = []
for cache_size in sorted(results.keys()):
    row = {"Cache_Size": cache_size}
    row.update(results[cache_size])
    df_rows.append(row)

df = pd.DataFrame(df_rows)

# Specify column order
columns = ["Cache_Size"]
for name in cache_strategies:
    name = name.upper()
    columns += [f"{name}_hitrate"]
for name in cache_strategies:
    name = name.upper()
    columns += [f"{name}_time_total"]
for name in cache_strategies:
    name = name.upper()
    columns += [f"{name}_hitrate_after_cold"]
for name in cache_strategies:
    name = name.upper()
    columns += [f"{name}_time_after_cold"]

df = df[columns]

# Save or print results
df.to_csv("./test/summary.csv", index=False)
print(df)
