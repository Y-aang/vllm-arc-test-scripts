import time
import threading

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer
)

model_name = "mistralai/mistral-7b-v0.1"  # Replace with your model
base_text = "This is a sentence of 32 tokens, please follow this prompt to generate some sentences. You can generate any thing that you want. Best luck, guy"
repeat_count = 64 // 32
document = (base_text * repeat_count).strip().rsplit(" ", 1)[0]
prompt = document

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use half precision if supported
    device_map="auto"          # Automatically place on GPU
)

# -------------------------------------------------------------------
# 1. Create a TextIteratorStreamer instance
#    skip_prompt=True skips the original input portion in the output
#    skip_special_tokens=True filters out special tokens
# -------------------------------------------------------------------
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# -------------------------------------------------------------------
# 2. Define a thread to consume the streamer's output. Each time the
#    streamer has a new text fragment, it will be processed in this thread.
# -------------------------------------------------------------------
start_time = None
first_token_time = None

def stream_consumer():
    global first_token_time
    for new_text in streamer:
        # If this is the first generated content received, record the first token arrival time
        if first_token_time is None:
            first_token_time = time.time()
        print(new_text, end="", flush=True)

# -------------------------------------------------------------------
# 3. In the main thread, start the consumer thread first, then call generate()
# -------------------------------------------------------------------
consumer_thread = threading.Thread(target=stream_consumer)
consumer_thread.start()

# Record time before calling generate()
start_time = time.time()

# Disable KV Cache for demonstration only (significantly reduces inference speed)
model.config.use_cache = False

# Call generate() and pass in the streamer
input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
output_ids = model.generate(
    **input_ids,
    streamer=streamer,
    max_new_tokens=50,
    use_cache=False  # Explicitly disable KV Cache again
)

consumer_thread.join()  # Wait for generation stream to finish

# -------------------------------------------------------------------
# 4. Calculate time to first token (TTF)
# -------------------------------------------------------------------
if first_token_time is not None:
    ttf = first_token_time - start_time
    print(f"\nTime to first token: {ttf:.4f} seconds")
else:
    print("No tokens were generated.")
