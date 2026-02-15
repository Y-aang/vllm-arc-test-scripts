from vllm import LLM, SamplingParams
import pickle

# Load data
with open("/home/shenyang/data/doc_query_dict.pkl", 'rb') as pfile:
    doc_query_dict = pickle.load(pfile)

print('Total documents:', len(doc_query_dict))

# Get the first document and its first query
first_doc, first_queries = next(iter(doc_query_dict.items()))
first_query = first_queries[0]

# Construct the prompt
prompt = f"Wiki Document: {first_doc}\nAnswer the following question based on WikiQA:\nQuestion: {first_query}\nAnswer:"
print("\n=== DEBUG: Generated Prompt ===\n")
print(prompt)

# Initialize vLLM
model_name = "mistralai/mistral-7b-v0.1"
llm = LLM(model=model_name, 
          max_model_len=4096,
          dtype="float16",
          enable_prefix_caching=True,
          disable_sliding_window=True,
        )

# Run inference
sampling_params = SamplingParams(max_tokens=15)
output = llm.generate(prompt, sampling_params)

# Print output
print("\n=== DEBUG: Model Output ===\n")
print(output[0].outputs[0].text.strip())
