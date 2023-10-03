import json
import random

# Specify the number of random elements you want to pick
sample_size = 200

# Specify the path of the input JSON file
input_file_path = 'parsed_siirtokarjalaiset.json'

# Specify the path of the output JSON file
output_file_path = 'sample2_siirtokarjalaiset.json'

# Step 1: Load JSON from the input file
with open(input_file_path, 'r') as infile:
    data = json.load(infile)

# Check if the sample size is greater than the total number of items
if sample_size > len(data):
    raise ValueError("Sample size must be less than or equal to the number of items in the dataset")

# Step 2: Pick random elements from the loaded data
sampled_data = random.sample(data, sample_size)

# Step 3: Recount the indexes
for index, item in enumerate(sampled_data, start=1):
    item['index'] = index

# Step 4: Write the sampled data to the output file
with open(output_file_path, 'w') as outfile:
    json.dump(sampled_data, outfile, indent=4)

print(f"{sample_size} random elements have been written to {output_file_path}")
