import json
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm
import random
import os
import requests
# os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

random.seed(42)
np.random.seed(42)

# Initialize vLLM
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
llm = LLM(model=model_name, 
          gpu_memory_utilization=0.99,
          max_model_len=1024, 
          block_size=16, 
          disable_sliding_window=True, 
          enable_prefix_caching=True)
tokenizer = llm.get_tokenizer()

def get_data():
    # Load QuALITY dataset
    splits = {
        "train": "https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.train",
        "dev": "https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev",
        "test": "https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.test"
    }

    grouped_data = {}
    error_lines = []

    for split_name, url in splits.items():
        print(f"Loading {split_name}...")
        r = requests.get(url)
        r.raise_for_status()
        lines = r.text.strip().splitlines()

        for idx, line in enumerate(lines):
            try:
                sample = json.loads(line)
                article = sample["article"].strip()
                question_texts = [q["question"] for q in sample["questions"]]

                if article not in grouped_data:
                    grouped_data[article] = []
                grouped_data[article].extend(question_texts)
            except json.JSONDecodeError as e:
                print(f"[{split_name}] JSONDecodeError at line {idx}: {e}")
                error_lines.append((split_name, idx))

    # Select different contexts (keys)
    selected_texts = list(grouped_data.keys())
    # selected_texts = selected_texts[:1000]  # No longer random
    selected_texts = selected_texts  # No longer random
    return grouped_data, selected_texts
    
grouped_data, selected_texts = get_data()

# Function: calculate the average length of text + question
def calculate_average_dq_length(grouped_data, selected_texts, tokenizer):
    dq_lengths = []
    text_lengths = []
    print("\nCalculating average length of doc + question (tokens):")
    for text in tqdm(selected_texts, desc="Processing DQ pairs", unit=" context"):
        text_tokens = tokenizer(text, return_tensors="pt")["input_ids"].shape[1]
        text_lengths.append(text_tokens)
        for question in grouped_data[text]:
            question_tokens = tokenizer(question, return_tensors="pt")["input_ids"].shape[1]
            dq_lengths.append(text_tokens + question_tokens)

    # Calculate average length
    avg_dq_length = sum(dq_lengths) / len(dq_lengths)
    max_dq_length = max(dq_lengths)
    min_dq_length = min(dq_lengths)
    print(f"\nAverage Text + Question length (tokens): {avg_dq_length:.2f}")
    print(f"Max Text + Question length (tokens): {max_dq_length}")
    print(f"Min Text + Question length (tokens): {min_dq_length}")
    print(f"Number of texts: {len(dq_lengths)}")

    # Count the number of questions for each context
    question_counts_list = [len(grouped_data[text]) for text in selected_texts]
    print("\nQuestion counts for each context in selected_texts:")
    print(question_counts_list)
    
    print('text lengths:', text_tokens)

    return avg_dq_length

# Call function to calculate average length
calculate_average_dq_length(grouped_data, selected_texts, tokenizer)

# Sample texts using power_law_with_hotspot      total_length=8000
def power_law_with_hotspot(data, total_length=2500, exponent=1.0, 
                           window_size=50, hotspot_ratio=0.10, hotspot_boost=10):
    num_windows = total_length // window_size
    result = []
    num_elements = len(data)
    values = np.arange(1, num_elements + 1)
    base_prob = values ** -exponent
    base_prob /= base_prob.sum()

    for _ in range(num_windows):
        hotspot_indices = np.random.choice(num_elements, size=max(1, int(hotspot_ratio * window_size)), replace=False)
        prob = base_prob.copy()
        prob[hotspot_indices] *= hotspot_boost
        prob /= prob.sum()
        
        sampled_indices = np.random.choice(num_elements, size=window_size, p=prob)
        result.extend([data[i] for i in sampled_indices])

    return result

def distribution_shift_sampling(data, total_length=1024*3, window_size=512, 
                                exponent=1.0, shuffle_each_window=True):
    num_windows = total_length // window_size
    result = []

    for _ in range(num_windows):
        # Shuffle once per window
        if shuffle_each_window:
            np.random.shuffle(data)

        num_elements = len(data)
        values = np.arange(1, num_elements + 1)
        base_prob = values ** -exponent
        base_prob /= base_prob.sum()
        
        # Sample within the current window
        sampled_indices = np.random.choice(len(data), size=window_size, p=base_prob)
        result.extend([data[i] for i in sampled_indices])

    return result


# sampled_texts = power_law_with_hotspot(selected_texts)
sampled_texts = distribution_shift_sampling(selected_texts)

# Function: append a random question to each sampled text
def append_random_question(sampled_texts, grouped_data):
    appended_texts = []
    for text in sampled_texts:
        if text in grouped_data and grouped_data[text]:
            chosen_question = random.choice(grouped_data[text])
            appended_texts.append(text + " " + chosen_question)
        else:
            assert False
            appended_texts.append(text)  # Keep original text if no questions available
    return appended_texts

# Use the function to append a random question to each element of sampled_texts
appended_texts = append_random_question(sampled_texts, grouped_data)

# Print first 10 examples
print("\nSampled Texts with Appended Questions (First 10):")
for i, text in enumerate(appended_texts[:1]):
    print(f"{i + 1}: {text}\n")

# Optional: save results as JSON (for debugging)
json_path = "/home/shenyang/tests/synthesis/Quality/sample/quality_sampled_texts_with_questions.json"
with open(json_path, 'w', encoding='utf-8') as json_file:
    json.dump(appended_texts, json_file, ensure_ascii=False, indent=4)

print(f"\nSaved to: {json_path}, generated {len(appended_texts)} samples.")
