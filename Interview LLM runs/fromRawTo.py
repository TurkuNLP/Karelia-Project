import json
import os
import re

def clean_and_sort_data(folder_path):
    cleaned_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            try:
                batch_number = int(re.search(r'batch_(\d+)', filename).group(1))
            except AttributeError:
                # If the filename doesn't match the expected pattern, skip it
                continue

            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as infile:
                try:
                    data = json.load(infile)
                    # Navigate to the nested 'response' key
                    response_data = data['response']
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from {filename}")
                    continue
                except KeyError:
                    print(f"'response' key not found in {filename}")
                    continue

                # Locate the specific JSON structure and capture everything after
                try:
                    end_of_interest_index = response_data.find('"}') + 2  # Finds the end of the specified JSON structure
                    remaining_content = response_data[end_of_interest_index:]
                    cleaned_content = remaining_content.strip()

                    cleaned_data.append({
                        "batch_number": batch_number,
                        "person_index": batch_number,
                        "api_response": cleaned_content
                    })
                except ValueError:
                    print(f"Unable to locate end of specified JSON structure in {filename}")
                    continue

    # Sort the list by 'batch_number'
    sorted_data = sorted(cleaned_data, key=lambda x: x['batch_number'])

    return sorted_data

def write_sorted_data_to_file(sorted_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(sorted_data, outfile, indent=4, ensure_ascii=False)

# Usage
folder_path = 'raw_api_responses3/config_True_1_1.0_0_0.9_400'  # Replace with the path to your folder
output_file = 'fromRawToAll_run3_config_True_1_1.0_0_0.9_400.json'  # Replace with the path to your desired output file

sorted_data = clean_and_sort_data(folder_path)
write_sorted_data_to_file(sorted_data, output_file)
