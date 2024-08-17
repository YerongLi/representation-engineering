
model_name_or_path = "/home/yerong2/models/internlm-xcomposer2d5-7b"
# model_name_or_path = 'internlm/internlm-xcomposer2d5-7b'

import torch
from transformers import AutoModel, AutoTokenizer


model = AutoModel.from_pretrained(model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True).half().eval().cuda()

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model.tokenizer = tokenizer

query = 'Image1 <ImageHere>; Image2 <ImageHere>; Image3 <ImageHere>; I want to buy a car from the three given cars, analyze their advantages and weaknesses one by one'
image = ['./examples/cars1.jpg',
        './examples/cars2.jpg',
        './examples/cars3.jpg',]
with torch.autocast(device_type='cuda', dtype=torch.float16):
    response, his = model.chat(tokenizer, query, image, do_sample=False, num_beams=3, use_meta=True)
print(response)

