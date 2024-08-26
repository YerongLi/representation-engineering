import json
from collections import defaultdict
import os

# Get the path from the environment variable
datasets_path = os.getenv('DATASETS')
# Define input and output file paths
input_file_path = datasets_path+'/MathV360K/train_samples_all_tuning.json'
output_file_path = 'train.json'

# Load the original data
with open(input_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Define the allowed datasets as a set
allowed_datasets = {"PlotQA", "TabMWP", "ChartQA","IconQA", "Geometry3K", "GeoQA+", "UniGeo", "TQA", "AI2D", "ScienceQA", "MapQA", "DVQA"}

# Prepare the new format and count datasets
new_data = []
dataset_counts = defaultdict(int)
total_count = 0

for i, entry in enumerate(data):
    if i> 38733: break
    dataset_name = entry['image'].split('/')[0]  # Extract dataset name
    # Check if the dataset name is in the allowed datasets
    if dataset_name in allowed_datasets:
        # Create conversation entries for human and bot
        conversations = [
            {
                "from": "human",
                "value": entry['conversations'][0]['value'].replace('<image>', '<ImageHere>')  # First message from "human"
            },
            {
                "from": "bot",
                "value": entry['conversations'][1]['value']  # Second message from "bot"
            }
        ]
        
        # Create the new entry
        new_entry = {
            "id": int(entry['id'].split('_')[1]),  # Convert 'identity_X' to integer X
            "conversations": conversations,
            "image": [f"{datasets_path}/MathV360K/data_images/{entry['image']}"]
        }
        new_data.append(new_entry)
        dataset_counts[dataset_name] += 1
        total_count += 1

# Save the new formatted data
with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=4)

# Print dataset counts and total count
print(f"Data has been successfully reformatted and saved to '{output_file_path}'.")
print("Dataset counts:")
for dataset, count in dataset_counts.items():
    print(f"{dataset}: {count}")
print(f"Total count: {total_count}")

# =============================================== #

input_file_path = datasets_path+'/MathV360K/trainsamples_qsa_tuning.json'
output_file_path = 'trainCoT.json'

# Load the original data
with open(input_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Define the allowed datasets as a set
allowed_datasets = {"PlotQA", "TabMWP", "ChartQA","IconQA", "Geometry3K", "GeoQA+", "UniGeo", "TQA", "AI2D", "ScienceQA", "MapQA", "DVQA"}

# Prepare the new format and count datasets
new_data = []
dataset_counts = defaultdict(int)
total_count = 0

for i, entry in enumerate(data):
    if i> 38733: break
    dataset_name = entry['image'].split('/')[0]  # Extract dataset name
    # Check if the dataset name is in the allowed datasets
    if dataset_name in allowed_datasets:
        # Create conversation entries for human and bot
        conversations = [
            {
                "from": "human",
                "value": entry['conversations'][0]['value'].replace('<image>', '<ImageHere>')  # First message from "human"
            },
            {
                "from": "bot",
                "value": entry['conversations'][1]['value']  # Second message from "bot"
            }
        ]
        
        # Create the new entry
        new_entry = {
            "id": int(entry['id'].split('_')[1]),  # Convert 'identity_X' to integer X
            "conversations": conversations,
            "image": [f"{datasets_path}/MathV360K/data_images/{entry['image']}"]
        }
        new_data.append(new_entry)
        dataset_counts[dataset_name] += 1
        total_count += 1

# Save the new formatted data
with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=4)

# Print dataset counts and total count
print(f"Data has been successfully reformatted and saved to '{output_file_path}'.")
print("Dataset counts:")
for dataset, count in dataset_counts.items():
    print(f"{dataset}: {count}")
print(f"Total count: {total_count}")