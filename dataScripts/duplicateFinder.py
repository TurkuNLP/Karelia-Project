import json

# Load the data
with open('yhdistetty_henkilotiedot.json', 'r') as file:
    data = json.load(file)

# Track entries by 'info' to identify duplicates
info_dict = {}
for entry in data:
    info_content = entry['info']
    if info_content in info_dict:
        info_dict[info_content].append(entry)
    else:
        info_dict[info_content] = [entry]

# Extract entries where 'info' section is duplicated
duplicates = [entries for info, entries in info_dict.items() if len(entries) > 1]

# Flatten the list of lists for saving to file
duplicate_entries = [entry for group in duplicates for entry in group]

# Save the duplicates next to each other in a file
with open('duplicates.json', 'w') as file:
    json.dump(duplicate_entries, file, indent=4)
