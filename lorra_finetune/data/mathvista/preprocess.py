import json

# Define the output file name as a variable
OUTPUTFILE = 'directmini.json'

# Load the JSON data from testmini.json
with open('testmini.json', 'r') as file:
    data = json.load(file)

# Initialize an empty list to hold the reformatted data
reformatted_data = []

# Iterate over each item in the data
for idx, (key, value) in enumerate(data.items()):
    # Extract the question and answer
    question = value.get('question', '')
    answer = value.get('answer', '')
    image_path = value.get('image', '')

    # Create the new format for each item
    conversation = {
        "id": idx,
        "conversations": [
            {"from": "human", "value": f" <ImageHere> {question}"},
            {"from": "bot", "value":  f"This answer is {answer}."}
        ],
        "image": f"data/mathvista/{image_path}"
    }
    
    # Append to the reformatted data list
    reformatted_data.append(conversation)

# Dump the reformatted data to the specified output file
with open(OUTPUTFILE, 'w') as file:
    json.dump(reformatted_data, file, indent=4)

print(f"Reformatted data has been written to {OUTPUTFILE}")
