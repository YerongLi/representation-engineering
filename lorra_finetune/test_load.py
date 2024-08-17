import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig

# Load the base model and tokenizer
model_path = "/home/yerong2/models/internlm-xcomposer2d5-7b"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load the adapter/finetuned weights
checkpoint_path = "lorra_finetune/src/finetune/output/finetune_lora/checkpoint-2/"
adapter_weights = torch.load(f"{checkpoint_path}/adapter_model.bin")

model = PeftModel.from_pretrained(model, checkpoint_path)
model = model.merge_and_unload()
# Verify that the model has loaded the weights
print("Model successfully loaded with finetuned weights.")

model.tokenizer = tokenizer

query = 'Image1 <ImageHere>; Image2 <ImageHere>; Image3 <ImageHere>; I want to buy a car from the three given cars, analyze their advantages and weaknesses one by one'
image = ['../examples/cars1.jpg',
        '../examples/cars2.jpg',
        '../examples/cars3.jpg',]
with torch.autocast(device_type='cuda', dtype=torch.float16):
    response, his = model.chat(tokenizer, query, image, do_sample=False, num_beams=3, use_meta=True)
print(response)