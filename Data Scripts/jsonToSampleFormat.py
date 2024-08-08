import json

# Specify the path of the input JSON file
input_file_path = 'sample2_siirtokarjalaiset.json'

# Specify the path of the output JSON file
output_file_path = 'sample2_siirtokarjalaiset_format.json'

# Step 1: Load JSON from the input file
with open(input_file_path, 'r') as infile:
    data = json.load(infile)

# Initialize an empty list to hold the modified data
modified_data = []

# Step 2: Modify the loaded data
for item in data:
    modified_item = {
        "index": item["index"],
        "primary_person_name": item.get("primary_person_name", ""),
        "primary_person_id": item.get("primary_person_id", ""), # Added line
        "spouse_name": item.get("spouse_name", ""),
        "spouse_id": item.get("spouse_id", ""), # Added line
        "source_text": item.get("source_text", ""),
        "person_hobbies": "",
        "person_social_orgs": "",
        "spouse_hobbies": "",
        "spouse_social_orgs": "",
    }
    modified_data.append(modified_item)


# Step 3: Write the modified data to the output file
with open(output_file_path, 'w') as outfile:
    json.dump(modified_data, outfile, indent=4)

print(f"Modified JSON data has been written to {output_file_path}")
