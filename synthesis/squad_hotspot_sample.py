import json
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm
import random
import os
# os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

random.seed(42)
np.random.seed(42)

# Initialize vLLM
# model_name = "mistralai/mistral-7b-v0.1"
# model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
# model_name = "microsoft/Phi-3-mini-4k-instruct"
# model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
llm = LLM(model=model_name, 
          gpu_memory_utilization=0.95,
          max_model_len=1024, 
          block_size=16, 
          disable_sliding_window=True, 
          enable_prefix_caching=True)
tokenizer = llm.get_tokenizer()

# Load SQuAD dataset
dataset = load_dataset("rajpurkar/squad")
valid_data = dataset['validation']

# Group and select all texts, keeping all questions for each text
grouped_data = {}
for item in valid_data:
    text = item['context']
    if text not in grouped_data:
        grouped_data[text] = []
    grouped_data[text].append(item['question'])

# Select different contexts (keys)
selected_texts = list(grouped_data.keys())
# selected_texts = selected_texts[:1000]  # No longer random
selected_texts = selected_texts  # No longer random

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

# Sample texts using power_law_with_hotspot
def power_law_with_hotspot(data, total_length=8000, exponent=1.0, 
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

sampled_texts = power_law_with_hotspot(selected_texts)

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
json_path = "/home/shenyang/tests/synthesis/sample/squad_sampled_texts_with_questions.json"
with open(json_path, 'w', encoding='utf-8') as json_file:
    json.dump(appended_texts, json_file, ensure_ascii=False, indent=4)

print(f"\nSaved to: {json_path}, generated {len(appended_texts)} samples.")
