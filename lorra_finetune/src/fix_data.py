import json

# Load the JSON file
with open('data/single_turn_single_image_example.json', 'r') as file:
    data = json.load(file)

# Update the "image" value to be a singleton list
for item in data:
    if isinstance(item.get('image'), str):
        item['image'] = [item['image']]

# Save the modified JSON back to the file
with open('data/single_turn_single_image_example.json', 'w') as file:
    json.dump(data, file, indent=4)

print("JSON file updated successfully.")
