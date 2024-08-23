from transformers import AutoModel

# Define the model name or path
# model_name = "/scratch/bbrz/yirenl2/models/Llama-2-7b-chat-hf"
model_name = '/scratch/bbrz/yirenl2/models/internlm-xcomposer2d5-7b'
# Load the model
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# Access the model configuration
config = model.config

# Get the number of layers
num_layers = config.num_hidden_layers

print(f"The number of layers in the {model_name} model is: {num_layers}")

