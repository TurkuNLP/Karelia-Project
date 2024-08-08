import json
import os
import re

def process_api_responses(folder_path, output_file):
    with open(output_file, 'w') as outfile:
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                # Extracting batch number from the file name
                batch_number = int(re.search(r'raw_response_(\d+)', filename).group(1))

                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as infile:
                    data = json.load(infile)
                    content = data["choices"][0]["message"]["content"]

                    # Extracting and reordering content
                    def extract(pattern):
                        match = re.search(pattern, content)
                        return match.group(1).strip() if match else ""

                    person_name = extract(r'^(.*?):')
                    spouse_name = extract(r'\n\n(.*?):')

                    person_id = extract(r'PersonID: (.*?)\n')
                    spouse_id = extract(r'SpouseID: (.*?)\n')

                    person_hobbies = extract(r'PersonHobbies: (.*?)\n')
                    person_social_orgs = extract(r'PersonSocialOrgs: (.*?)\n')
                    spouse_hobbies = extract(r'SpouseHobbies: (.*?)\n')
                    spouse_social_orgs = extract(r'SpouseSocialOrgs: (.*?)$')

                    formatted_content_str = (
                        f'PersonID: {person_id}\n'
                        f'PersonName: {person_name}\n'
                        f'PersonHobbies: {person_hobbies}\n'
                        f'PersonSocialOrgs: {person_social_orgs}\n'
                        f'SpouseID: {spouse_id}\n'
                        f'SpouseName: {spouse_name}\n'
                        f'SpouseHobbies: {spouse_hobbies}\n'
                        f'SpouseSocialOrgs: {spouse_social_orgs}'
                    )

                    formatted_content = {
                        "batch_number": batch_number,
                        "person_index": batch_number,
                        "api_response": formatted_content_str
                    }

                    outfile.write(json.dumps(formatted_content) + "\n")


# Usage
folder_path = '/home/joonatan/Documents/Siirtokarjalaiset/siirtokarjalaisetTietokantaJson/apiResponse/raw_api_responses'  # Replace with the path to your folder
output_file = 'fromRawToAll_200_sample.json'  # Replace with the path to your desired output file


def convert_to_json_array(input_file, output_file):
    # Reading and parsing the JSON lines
    with open(input_file, 'r') as infile:
        data = [json.loads(line) for line in infile]

    # Sorting the entries based on batch_number
    sorted_data = sorted(data, key=lambda x: x['batch_number'])

    # Writing the sorted data as a JSON array
    with open(output_file, 'w') as outfile:
        json.dump(sorted_data, outfile, indent=4)
        

# Replace with the path to your folder containing raw API responses
folder_path = 'apiResponse/raw_api_responses'  

# Intermediate file to store processed API responses
intermediate_file = 'fromRawToAll_200_sample.json'  

# Process the raw API responses and create a JSON lines file
process_api_responses(folder_path, intermediate_file)

# Final output file for the sorted JSON array
final_output_file = 'sorted_fromRawToAll_200_sample.json'

# Convert the created JSON lines file into a sorted JSON array
convert_to_json_array(intermediate_file, final_output_file)
