import json

# The list of filenames, in the order you want to concatenate them
filenames = [
    'siirtokarjalaiset_I.json', 
    'siirtokarjalaiset_II.json', 
    'siirtokarjalaiset_III.json', 
    'siirtokarjalaiset_IV.json'
]

# The master list to hold all the data from the different files
master_list = []

# Loop over each filename
for filename in filenames:
    try:
        # Open the file and load its JSON content
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Append the data from this file to the master list
        if isinstance(data, list):
            master_list.extend(data)
        else:  # If the JSON content is a dictionary, append it as a single item
            master_list.append(data)
    except FileNotFoundError:
        print(f"{filename} not found")
    except json.JSONDecodeError:
        print(f"Error decoding JSON in {filename}")

# Write the master list to a new JSON file
with open('combined_siirtokarjalaiset.json', 'w', encoding='utf-8') as f:
    json.dump(master_list, f, indent=4, ensure_ascii=False)

print(f"Data combined and written to combined_siirtokarjalaiset.json")
