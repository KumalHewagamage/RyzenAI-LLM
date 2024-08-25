import argparse
import logging
import time
import gc
import os
import builtins
import sys
from pathlib import Path
sys.path.append("../opt")

from transformers import set_seed, AutoTokenizer
import psutil
import onnx
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForCausalLM

CURRENT_DIR = Path(__file__).parent
config_file_path = CURRENT_DIR / "vaip_config.json"

set_seed(123)

dev = os.getenv("DEVICE")
if dev == "stx":
    p = psutil.Process()
    p.cpu_affinity([0, 1, 2, 3])

# Set implementation for aie target
builtins.impl = "v0"
builtins.quant_mode = "w8a8"

provider = "VitisAIExecutionProvider"
provider_options = {'config_file': str(config_file_path)}

# Define paths
path = r"C:\Users\K_ADMIN\Desktop\AMD GenAI\smoothquant\LLaMa_onnx_final_matmult"
og_model_dir = r"C:\Users\K_ADMIN\Desktop\AMD GenAI\HF Stuff\Llama-2-7b-chat-hf"

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 0
sess_options.inter_op_num_threads = 0
sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
sess_options.add_session_config_entry("session.inter_op.allow_spinning", "0")

# Load the quantized model
try:
    model_files = [f for f in os.listdir(path) if f.endswith(".onnx")]
    if not model_files:
        raise FileNotFoundError("No ONNX model file found in the specified directory.")

    model_file = model_files[0]  # Assuming there's only one model file
    model_path = os.path.join(path, model_file)

    print(f"Loading model from {model_path}...")
    model = ORTModelForCausalLM.from_pretrained(
        path,
        file_name=model_file,
        provider=provider,
        use_cache=True,
        use_io_binding=False,
        session_options=sess_options,
        provider_options=provider_options
    )
    tokenizer = AutoTokenizer.from_pretrained(og_model_dir)
except Exception as e:
    print(f"Error loading the model: {e}")
    sys.exit(1)

def decode_prompt(model, tokenizer, prompt, input_ids=None, max_new_tokens=30):
    if input_ids is None:
        print(f"prompt: {prompt}")
        start = time.time()
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=False)  # Adjust truncation as needed
        end = time.time()
    else:
        start, end = 0, 0

    print("Input Setup - Elapsed Time: " + str(end - start))

    prompt_tokens = 0
    if prompt is None:
        start = time.time()
        generate_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, attention_mask=inputs['attention_mask'])
        end = time.time()
        prompt_tokens = input_ids.shape[1]
    else:
        start = time.time()
        generate_ids = model.generate(inputs.input_ids, max_length=100, pad_token_id=tokenizer.pad_token_id, attention_mask=inputs['attention_mask'])
        end = time.time()
        prompt_tokens = inputs.input_ids.shape[1]

    num_tokens_out = generate_ids.shape[1]
    new_tokens_generated = num_tokens_out - prompt_tokens
    generate_time = (end - start)
    time_per_token = (generate_time / new_tokens_generated) * 1e3
    print("Generation - Elapsed Time: " + str(generate_time))

    start = time.time()
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    end = time.time()

    print(f"response: {response}")
    #print("Tokenizer Decode - Elapsed Time: " + str(end - start))

def main():
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "quit", "q"]:
            print("Exiting chat...")
            break
        decode_prompt(model, tokenizer, prompt)

if __name__ == "__main__":
    # Ensure tokenizer has a padding token defined
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # You can replace '[PAD]' with any appropriate token

    main()
