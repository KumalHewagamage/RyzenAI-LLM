import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
)
from transformers import LlamaTokenizer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_llama_like
import tqdm

model_dir = r"C:\Users\K_ADMIN\Desktop\AMD GenAI\HF Stuff\Llama-2-7b-chat-hf"

model = LlamaForCausalLM.from_pretrained(
    model_dir, torch_dtype=torch.float16, device_map="auto"
)

act_scale_dir=r'C:\Users\K_ADMIN\Desktop\AMD GenAI\smoothquant\act_scales\llama-2-7b-chat-w4-g128.pt'
act_scales = torch.load(act_scale_dir,map_location=torch.device('cpu'))

smooth_lm(model, act_scales, 0.85)
model_smoothquant_w8a8 = quantize_llama_like(model)
print(model_smoothquant_w8a8)


torch.save(model_smoothquant_w8a8.state_dict(), 'pytorch_model.bin')

print("Model saved as 'pytorch_model.bin' successfully.")