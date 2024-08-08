import json

# Load the data
with open('sample2_siirtokarjalaiset_format.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Save the data back, ensuring that Unicode characters are preserved
with open('sample2_siirtokarjalaiset_unicode.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
