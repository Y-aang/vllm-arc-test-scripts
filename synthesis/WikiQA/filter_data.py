import pickle
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Initialize LLM and tokenizer (refer to download_wikiqa.py for usage)
llm = LLM(model="meta-llama/Llama-3.2-3B", 
          gpu_memory_utilization=0.9,
          max_model_len=2000, 
          block_size=16, 
          disable_sliding_window=True, 
          enable_prefix_caching=True
        )
tokenizer = llm.get_tokenizer()

# Read pickle file (refer to view_pickle.py for usage)
data_file = "wikiqa_doc_query_dict.pkl"
# data_file = "/home/shenyang/data/doc_query_dict.pkl"
with open(data_file, "rb") as pfile:
    doc_query_dict = pickle.load(pfile)

# Iterate over all keys (passages) and calculate token length for each passage
token_lengths = []
passage_token_pairs = []  # Store (passage, token_length) pairs
for passage in tqdm(doc_query_dict.keys()):
    # Calculate token length (refer to download_wikiqa.py for method)
    num_tokens = tokenizer(passage, return_tensors="pt")["input_ids"].shape[1]
    token_lengths.append(num_tokens)
    passage_token_pairs.append((passage, num_tokens))

# Calculate average
if token_lengths:
    avg_length = sum(token_lengths) / len(token_lengths)
    print(f"Total number of passages: {len(token_lengths)}")
    print(f"Average token length: {avg_length:.2f}")
    print(f"Minimum token length: {min(token_lengths)}")
    print(f"Maximum token length: {max(token_lengths)}")
    
    # Find the shortest wiki
    min_token_length = min(token_lengths)
    shortest_passage = min(passage_token_pairs, key=lambda x: x[1])[0]
    print(f"\nShortest wiki (Token length: {min_token_length}):")
    print("=" * 80)
    print(shortest_passage)
    print("=" * 80)
else:
    print("WARNING: No passage data found")

