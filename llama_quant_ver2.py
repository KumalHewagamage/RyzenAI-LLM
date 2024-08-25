import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import os 
from smoothquant.smooth import smooth_lm
import string
import random 

def convert_to_onnx(model_name):
    model_dir = r"C:\Users\K_ADMIN\Desktop\AMD GenAI\HF Stuff\Llama-2-7b-chat-hf"

    model = LlamaForCausalLM.from_pretrained(
    model_dir, torch_dtype=torch.float16, device_map="cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.tokenizer = tokenizer 
    
    # Replace with your data or prompt generation logic
    prompt = "What is the meaning of life"
    inputs = tokenizer(prompt, return_tensors="pt") 
    
    act_scale_dir=r'C:\Users\K_ADMIN\Desktop\AMD GenAI\smoothquant\act_scales\llama-2-7b-chat-w4-g128.pt'
    act_scales = torch.load(act_scale_dir,map_location=torch.device('cpu'))
    smooth_lm(model, act_scales, 0.5)
    print(model)
    
    model_out = model(inputs.input_ids)
    print(f"{(model_out.logits.shape)=}")
    
    out_dir = f"./{model_name}_smoothquant"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model.save_pretrained(os.path.join(out_dir, "pytorch"))

    onnx_export_dir = os.path.join(out_dir, "onnx")
    if not os.path.exists(onnx_export_dir):
        os.makedirs(onnx_export_dir)
    
    model.export(
        export_dir=onnx_export_dir,
        task="text-generation-with-past",
        framework="pt",
        save_optimized=True,
        postprocess=False,
    )

if __name__ == "__main__":
    convert_to_onnx("Llama2")
