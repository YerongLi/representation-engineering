import json
from collections import defaultdict
import os

# Get the path from the environment variable
datasets_path = os.getenv('DATASETS')

def process_data(input_file, output_file, output_folder, image_tag):
    # Define input and output file paths
    input_file_path = os.path.join(datasets_path, 'MathV360K', input_file)
    output_file_path = os.path.join(output_folder, output_file)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the original data
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Define the allowed datasets as a set
    allowed_datasets = {"PlotQA", "TabMWP", "ChartQA", "IconQA", "Geometry3K", "GeoQA+", "UniGeo", "TQA", "AI2D", "ScienceQA", "MapQA", "DVQA"}
    # Chart Geometry Table
    # IconQA
    m = {
        "PlotQA": "chart",
        "TabMWP": "table",
        "ChartQA": "chart",
        "IconQA": "default",
        "Geometry3K": "geometry",
        "GeoQA+": "geometry",
        "UniGeo": "geometry",
        "TQA": "visual",
        "AI2D": "visual",
        "ScienceQA": "visual",
        "MapQA": "visual",
        "DVQA": "default"
    }
    # Prepare the new format and count datasets
    new_data = []
    dataset_counts = defaultdict(int)
    total_count = 0

    # # Iterate over the original data
    # for i, entry in enumerate(data):
    #     if i > 38733:  # Limit processing to the first 38733 entries
    #         break
        
    #     dataset_name = entry['image'].split('/')[0]  # Extract dataset name
        
    #     # Check if the dataset name is in the allowed datasets
    #     if dataset_name in allowed_datasets:
    #         # Prepare the new entry in the desired format
    #         new_entry = {
    #             'type': 'image',
    #             "query": entry['conversations'][0]['value'].replace('<image>', image_tag),
    #             "response": entry['conversations'][1]['value'],
    #             "images": f"{datasets_path}/MathV360K/data_images/{entry['image']}",
    #             'system': m[dataset_name],
    #         }
            
    #         # Append the new entry to the new_data list
    #         new_data.append(new_entry)
            
    #         # Increment dataset count and total count
    #         dataset_counts[dataset_name] += 1
    #         total_count += 1

    # Prepare the new format and count datasets
    new_data = []
    dataset_counts = defaultdict(int)
    total_count = 0
    
    # Iterate over the original data
    for i, entry in enumerate(data):
        if i > 38733:  # Limit processing to the first 38733 entries
            break
        
        dataset_name = entry['image'].split('/')[0]  # Extract dataset name
        
        # Check if the dataset name is in the allowed datasets
        if dataset_name in allowed_datasets:
            # Prepare the new entry in the desired format
            new_entry = {
                'type': 'image',
                "query": entry['conversations'][0]['value'].replace('<image>', image_tag),
                "response": entry['conversations'][1]['value'],
                "images": f"{datasets_path}/MathV360K/data_images/{entry['image']}",
                'system': m[dataset_name],
            }
            
            # Append the new entry to the new_data list
            new_data.append(new_entry)
            
            # Increment dataset count and total count
            dataset_counts[m[dataset_name]] += 1
            total_count += 1
    
    # Print dataset counts and total count
    print("Dataset counts:")
    for dataset_type, count in dataset_counts.items():
        print(f"{dataset_type}: {count}")
    print(f"Total count: {total_count}")

    # Save the newly formatted data
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(new_data, file, ensure_ascii=False, indent=4)

    # Print dataset counts and total count
    print(f"Data has been successfully reformatted and saved to '{output_file_path}'.")
    print("Dataset counts:")
    for dataset, count in dataset_counts.items():
        print(f"{dataset}: {count}")
    print(f"Total count: {total_count}")


def process2(input_file, output_file, output_folder):
    # Define input and output file paths
    input_file_path = os.path.join(datasets_path, 'MathV360K', input_file)
    output_file_path = os.path.join(output_folder, output_file)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the original data
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Define the allowed datasets as a set
    allowed_datasets = {"PlotQA", "TabMWP", "ChartQA", "IconQA", "Geometry3K", "GeoQA+", "UniGeo", "TQA", "AI2D", "ScienceQA", "MapQA", "DVQA"}
    m = {
        "PlotQA": "chart",
        "TabMWP": "table",
        "ChartQA": "chart",
        "IconQA": "default",
        "Geometry3K": "geometry",
        "GeoQA+": "geometry",
        "UniGeo": "geometry",
        "TQA": "visual",
        "AI2D": "visual",
        "ScienceQA": "visual",
        "MapQA": "visual",
        "DVQA": "default"
    }
    # Prepare the new format and count datasets
    new_data = []
    dataset_counts = defaultdict(int)
    total_count = 0

    for i, entry in enumerate(data):
        if i > 38733: break
        dataset_name = entry['image'].split('/')[0]  # Extract dataset name
        # Check if the dataset name is in the allowed datasets
        if dataset_name in allowed_datasets:
            # Create the image tag with the full path
            img_tag = f"<img>{datasets_path}/MathV360K/data_images/{entry['image']}</img>"
            
            # Create conversation entries for user and assistant
            conversations = [
                {
                    "from": "user",
                    "value": img_tag + entry['conversations'][0]['value'].replace('<image>', '')  # User message with image tag
                },
                {
                    "from": "assistant",
                    "value": entry['conversations'][1]['value']  # Assistant message
                }
            ]
            
            # Create the new entry
            new_entry = {
                "conversations": conversations
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
# Process files for qw/ folder
process_data('train_samples_all_tuning.json', 'train.json', 'qw', '<image>')
process_data('trainsamples_qsa_tuning.json', 'trainCoT.json', 'qw', '<image>')

# process2('train_samples_all_tuning.json', 'train.json', 'it')
# process2('trainsamples_qsa_tuning.json', 'trainCoT.json', 'it')