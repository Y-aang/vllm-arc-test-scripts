from vllm import LLM, SamplingParams

# Use model name directly
model_name = "mistralai/Mistral-7B-v0.1"  # Replace with the model name you want to use

# Initialize LLM (vLLM handles model loading automatically)
llm = LLM(model=model_name, max_model_len=14096, block_size=256)

# Define inference parameters
sampling_params = SamplingParams(
    temperature=0.7,  # Controls generation randomness
    max_tokens=50,    # Maximum generation length
    top_p=0.9,        # Nucleus Sampling (Top-p)
)

# Input prompt
prompt = "Explain the importance of renewable energy in simple terms."

# Run inference
output = llm.generate(prompt, sampling_params=sampling_params)

# Print results
print("Input prompt:", prompt)
print("Generated output:", output[0])
