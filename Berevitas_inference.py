from transformers import AutoTokenizer
import torch
import numpy as np
from datasets import load_dataset
import onnxruntime as ort
import time



# load model
model_id = r"C:\Users\K_ADMIN\Desktop\AMD GenAI\OPT 1.3B\brevitas_opt_onnx"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize_function(examples):
    example = tokenizer(examples['text'])
    return example


# create session
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(r'C:\Users\K_ADMIN\Desktop\AMD GenAI\OPT 1.3B\brevitas_opt_onnx\model.onnx', options, providers=['VitisAIExecutionProvider'],provider_options=[{"config_file":"C:/Users/K_ADMIN/Desktop/AMD GenAI/GPT J/gpt-j-6B-int8-dynamic/vaip_config.json"}])
#session = ort.InferenceSession('C:/Users/K_ADMIN/Desktop/AMD GenAI/GPT J/gpt-j-6B-int8-dynamic/model.onnx', options, providers=['CPUExecutionProvider'])
#session = ort.InferenceSession('C:/Users/K_ADMIN/Desktop/AMD GenAI/GPT J/gpt-j-6B-int8-dynamic/model.onnx', providers=['VitisAIExecutionProvider'], provider_options=[{"config_file":"C:/Users/K_ADMIN/Desktop/AMD GenAI/GPT J/gpt-j-6B-int8-dynamic/vaip_config.json"}])
print("inference started")
# input prompt
# 32 tokens input
prompt = "Once upon a time, there existed a little girl, who liked to have adventures." + \
                 " She wanted to go to places and meet new people, and have fun."

print("prompt: ", prompt)

total_time = 0.0
num_iter = 4
num_warmup = 3

# start
for idx in range(num_iter):
    text = []
    tic = time.time()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    attention_mask = torch.ones(input_ids.shape[1] +1)
    attention_mask[0] = 0
    attention_mask = attention_mask.unsqueeze(0)

    inp = {'input_ids': input_ids.detach().cpu().numpy(),
            'attention_mask': attention_mask.detach().cpu().numpy().astype('int64')}
    for i in range(28):
        inp["past_key_values.{}.key".format(i)] = torch.zeros([1,16,1,256]).detach().cpu().numpy()
        inp["past_key_values.{}.value".format(i)] = torch.zeros([1,16,1,256]).detach().cpu().numpy()

    for step in range(32):

        output = session.run(None, inp)
        logits = output[0]
        logits = torch.from_numpy(logits)
        next_token_logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        next_tokens = torch.argmax(probs, dim=-1)
        present_kv = output[1]
        for i in range(28):

            if step == 0:
                inp["past_key_values.{}.key".format(i)] = output[2*i+1][:, :, 1:, :]
                inp["past_key_values.{}.value".format(i)] = output[2*i+2][:, :, 1:, :]
            else:
                inp["past_key_values.{}.key".format(i)] = output[2*i+1]
                inp["past_key_values.{}.value".format(i)] = output[2*i+2]

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if step == 0:
            attention_mask = torch.cat([attention_mask[:, 1:], torch.ones([1, 1])], dim=-1)
        else:
            attention_mask = torch.cat([attention_mask, torch.ones([1, 1])], dim=-1)

        inp['attention_mask'] = attention_mask.detach().cpu().numpy().astype('int64')
        inp['input_ids'] = input_ids[:, -1:].detach().cpu().numpy()

    print(tokenizer.decode(input_ids[0]))
    toc = time.time()
    if idx >= num_warmup:
        total_time += (toc - tic)
print("Inference latency: %.3f s." % (total_time / (num_iter - num_warmup)))