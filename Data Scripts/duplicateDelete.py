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

# Extract unique entries
unique_entries = [entries[0] for info, entries in info_dict.items() if len(entries) == 1]

# If you want to keep one of the duplicates and treat it as unique
for group in duplicates:
    unique_entries.append(group[0])

# Recount the indexes for the unique entries
for i, entry in enumerate(unique_entries):
    entry["index"] = i

# Save the unique and re-indexed entries to a new JSON file
with open('cleaned_yhdistetty_henkilotiedot.json', 'w') as file:
    json.dump(unique_entries, file, indent=4)
