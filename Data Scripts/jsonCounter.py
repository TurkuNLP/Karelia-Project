import json

# Load the data from the JSON file
with open('parsed_siirtokarjalaiset.json', 'r') as file:
    data = json.load(file)

# Count the number of elements
count = len(data)

print(f"There are {count} elements in the JSON file.")
