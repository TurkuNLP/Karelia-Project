import json

# Load JSON data
filename = 'combined_siirtokarjalaiset.json'
with open(filename, 'r') as f:
    data = json.load(f)

parsed_data = []

# Iterate over each person in the JSON array
for idx, person in enumerate(data):
    # Extract the required information from the JSON object
    primary_person = person.get('primaryPerson', {}) or {}
    primary_person_name = (primary_person.get('name', {}).get('firstNames') or '') + ' ' + \
                          (primary_person.get('name', {}).get('surname') or '')
    primary_person_id = primary_person.get('kairaId', '')
    
    spouse = person.get('spouse', {}) or {}
    spouse_name = (spouse.get('firstNames') or '') + ' ' + (spouse.get('formerSurname') or '')
    spouse_id = spouse.get('kairaId', '')
    
    source_text = person.get('personMetadata', {}).get('sourceText', '')
    
    # Append the extracted information to the parsed_data list
    parsed_data.append({
        'index': idx,
        'primary_person_name': primary_person_name,
        'primary_person_id': primary_person_id,
        'spouse_name': spouse_name,
        'spouse_id': spouse_id,
        'source_text': source_text
    })

# If you want to write the extracted data to a JSON file
with open('parsed_siirtokarjalaiset.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(parsed_data, indent=4, ensure_ascii=False))
