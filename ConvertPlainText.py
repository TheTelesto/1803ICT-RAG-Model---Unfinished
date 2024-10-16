import json

# Specify the path to your text file
file_path = 'Data\KeyTerms.txt'

# Open the text file and read its contents with UTF-8 encoding
with open(file_path, 'r', encoding='utf-8') as file:
    plain_text = file.read()

# Split the plain text into individual entries based on newlines
entries = plain_text.strip().split("\n")

# Create a list of dictionaries to structure the data
dataset = []
for entry in entries:
    entry = entry.strip()  # Remove any leading/trailing spaces
    if entry:  # Check if the entry is not empty
        if ":" in entry:
            # If the line contains a colon, split it into concept and definition
            concept, definition = entry.split(":", 1)
            dataset.append({"text": f"{concept.strip()}: {definition.strip()}"})
        else:
            # If the line does not contain a colon, treat the whole line as text
            dataset.append({"text": entry})

# Convert the list into a JSON file
with open('dataset.json', 'w') as f:
    json.dump(dataset, f, indent=4)

print("Dataset successfully created!")
