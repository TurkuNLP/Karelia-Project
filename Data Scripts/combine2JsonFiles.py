import json

def combine_json_files(file1, file2, output_file):
    # Read the first JSON file
    with open(file1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)

    # Read the second JSON file
    with open(file2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)

    # Combine the data from both files
    combined_data = data1 + data2

    # Write the combined data to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)

# Example usage
combine_json_files('apiResponse/All_responses10100.json', 'apiResponse/responses_10100_19000.json', 'combined_output.json')