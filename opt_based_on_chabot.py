# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.

import logging
import time
import gc
import os
import sys 
from ext.model_utils import warmup, decode_prompts, perplexity
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import set_seed, AutoTokenizer, OPTForCausalLM
import pathlib
from ext.smoothquant.smoothquant import smooth
import torch
import random 
import string

CURRENT_DIR = pathlib.Path(__file__).parent
print(CURRENT_DIR.parent)
config_file_path = CURRENT_DIR / "vaip_config.json"

# Direct input values
model_name = "opt-1.3b"
download = False
quantize = False
use_cache = True
local_path = r"C:\Users\K_ADMIN\Desktop\AMD GenAI\OPT 1.3B\opt-1.3b"
target = "aie"
disable_cache = True

set_seed(123)

log_dir = f"./logs_{model_name}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = f"{log_dir}/log_{model_name}_cpu.log"
logging.basicConfig(filename=log_file,
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.CRITICAL)

if download:        
    model = OPTForCausalLM.from_pretrained(f"facebook/{model_name}")
    out_dir = f"./{model_name}_pretrained_fp32"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model.save_pretrained(out_dir)
    print(f"Saving downloaded fp32 model...{out_dir}\n")

elif quantize:
    # Smooth quantize
    path = r"C:\Users\K_ADMIN\Desktop\AMD GenAI\OPT 1.3B\opt-1.3b"
    if not os.path.exists(path):
        print(f"Pretrained fp32 model not found, exiting..")
        sys.exit(1)
    model = OPTForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\K_ADMIN\Desktop\AMD GenAI\OPT 1.3B\opt-1.3b")
    model.tokenizer = tokenizer
    act_scales = torch.load(r"C:\Users\K_ADMIN\Desktop\AMD GenAI\OPT 1.3B\ext\smoothquant\act_scales\opt-1.3b.pt")
    smooth.smooth_lm(model, act_scales, 0.5)
    print(model)

    prompt = ''.join(random.choices(string.ascii_lowercase + " ", k=model.config.max_position_embeddings))
    inputs = tokenizer("What is meaning of life", return_tensors="pt") 
    print(f"inputs: {inputs}")
    print(f"inputs.input_ids: {inputs.input_ids}")
    for key in inputs.keys():
        print(inputs[key].shape)
        print(inputs[key])
    model_out = model(inputs.input_ids)
    print(f"{model_out.logits.shape=}")
    out_dir = f"./{model_name}_smoothquant"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model.save_pretrained(out_dir + "/model_onnx")
    print(f"Saving Smooth Quant fp32 model...\n")

    # Quantize to int8
    print(f"Quantizing model with Optiaum...\n")
    os.system(f'optimum-cli export onnx -m {model_name}_smoothquant/model_onnx --task text-generation-with-past {model_name}_smoothquant/model_onnx_int8 --framework pt --no-post-process')
    print(f"Saving quantized int8 model ...\n")

# Deploy and test model
else: 
    if target == "aie":
        provider = "VitisAIExecutionProvider"
        provider_options = {'config_file': str(config_file_path)} 
    else:
        provider = "CPUExecutionProvider"
        provider_options = {} 

    path = "facebook/"
    if local_path != "":
        path = local_path
    
    model = ORTModelForCausalLM.from_pretrained(r"C:\Users\K_ADMIN\Desktop\AMD GenAI\OPT 1.3B\opt-1.3b_smoothquant\model_onnx_int8", provider=provider, use_cache=disable_cache, use_io_binding=False, provider_options=provider_options)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

    collected = gc.collect()

    input_prompt ="what does pizza hut do"

    warmup(model, tokenizer, None,input_prompt)
    
    decode_prompts(model, tokenizer)
    logging.shutdown()
