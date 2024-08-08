import json

def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def extract_and_count(data, keys):
    count_dict = {}
    for entry in data:
        response = entry["api_response"]
        lines = response.split('\n')
        for line in lines:
            for key in keys:
                if key in line:
                    hobbies_or_orgs = line.split(": ")[1].split(", ")
                    for item in hobbies_or_orgs:
                        if item:
                            count_dict[item] = count_dict.get(item, 0) + 1
    return count_dict

def calculate_percentage(count_dict):
    total_count = sum(count_dict.values())
    return {item: {"count": count, "percentage": (count / total_count) * 100} 
            for item, count in count_dict.items()}

file_path = 'apiResponse/all_responses_5976.json'
data = read_data(file_path)

# Extracting and counting for combined categories
combined_hobbies_count = extract_and_count(data, ["PersonHobbies", "SpouseHobbies"])
combined_social_orgs_count = extract_and_count(data, ["PersonSocialOrgs", "SpouseSocialOrgs"])

# Sorting and calculating percentages
combined_hobbies_count = dict(sorted(combined_hobbies_count.items(), key=lambda item: item[1], reverse=True))
combined_social_orgs_count = dict(sorted(combined_social_orgs_count.items(), key=lambda item: item[1], reverse=True))


# Formatting data for readability
def format_readable(combined_data):
    readable_data = {}
    for category, items in combined_data.items():
        readable_data[category] = {item: f"{details['count']}, {details['percentage']:.2f}%" 
                                   for item, details in items.items()}
    return readable_data
# Combining count and percentage data
combined_data = {
    "CombinedHobbiesCount": calculate_percentage(combined_hobbies_count),
    "CombinedSocialOrgsCount": calculate_percentage(combined_social_orgs_count)
}



readable_combined_data = format_readable(combined_data)

# Writing the combined data to a JSON file
output_file_path = 'combinedHowManyHobbies2.json'
with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(readable_combined_data, file, ensure_ascii=False, indent=4)