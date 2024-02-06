from transformers import AutoTokenizer
import Levenshtein
import json
import re

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")

# Function to load JSON data
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to save data to JSON file
def save_to_json_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

# Function to reconstruct text from tokenized form
def reconstruct_text(tokenized_text):
    reconstructed = ""
    for token in tokenized_text:
        if token.startswith("##"):
            reconstructed += token[2:]
        else:
            reconstructed += ' ' + token
    return reconstructed.strip()

# Function to find the best matches for a phrase within a target text
def find_best_matches_char_basis(phrase, target_text, min_similarity_ratio):
    matches = []
    for start_idx in range(len(target_text)):
        for end_idx in range(start_idx + 1, len(target_text) + 1):
            substring = target_text[start_idx:end_idx]
            distance = Levenshtein.distance(phrase, substring)
            max_len = max(len(phrase), len(substring))
            ratio = 1 - (distance / max_len) if max_len > 0 else 0
            if ratio >= min_similarity_ratio:
                matches.append((substring, ratio, start_idx, end_idx))

    matches.sort(key=lambda x: x[1], reverse=True)  # Sort by ratio descending

    return matches


# Function to label matching sequences
def label_matching_sequence(tokenized_text, untokenized_string, labels, label_prefix, reconstructed_text):
    pattern = re.compile(r'\s*'.join(re.escape(word) for word in untokenized_string.split()), re.IGNORECASE)
    matches = pattern.finditer(reconstructed_text)
    for match in matches:
        start_idx, end_idx = match.span()
        print(f"Entity '{untokenized_string}' found at positions {start_idx} to {end_idx} in reconstructed text.")

        # Check if all tokens in the match are currently unlabeled ('o')
        all_unlabeled = True
        for i, token in enumerate(tokenized_text):
            token_start = len(reconstruct_text(tokenized_text[:i]))
            token_end = token_start + len(token.replace("##", ""))
            if token_start < end_idx and token_end > start_idx and labels[i] != 'o':
                all_unlabeled = False
                break

        # Apply labels if all tokens are unlabeled
        if all_unlabeled:
            found_start = False
            for i, token in enumerate(tokenized_text):
                token_start = len(reconstruct_text(tokenized_text[:i]))
                token_end = token_start + len(token.replace("##", ""))
                if token_start < end_idx and token_end > start_idx:
                    label = 'B-' + label_prefix if not found_start else 'I-' + label_prefix
                    found_start = True
                    labels[i] = label
                    print(f"   Token: '{token}' labeled as '{label}'")
            return labels  # Return after successful labeling

    print(f"No suitable match found for '{untokenized_string}' within confidence threshold.")
    return labels



# Main function to process data
def process_data(sourceData, apiResults, start_index, end_index, max_distance_threshold):
    processed_data = []
      
    for source_record in sourceData:
        source_index = source_record["index"]
        # Process records within the specified index range
        if start_index <= source_index <= end_index:
            primary_person_name = " ".join(word.capitalize() for word in source_record["primary_person_name"].split())
            source_text = primary_person_name + ", " + source_record["source_text"]


            api_record = apiResults[source_index]
            entities = [
                (person_hobby, "P-HOB") for person_hobby in api_record["api_response"].split("\nPersonHobbies: ")[1].split("\n")[0].split(", ")
            ] + [
                (spouse_hobby, "S-HOB") for spouse_hobby in api_record["api_response"].split("\nSpouseHobbies: ")[1].split("\n")[0].split(", ")
            ] + [
                (person_org, "P-ORG") for person_org in api_record["api_response"].split("\nPersonSocialOrgs: ")[1].split("\n")[0].split(", ")
            ] + [
                (spouse_org, "S-ORG") for spouse_org in api_record["api_response"].split("\nSpouseSocialOrgs: ")[1].split("\n")[0].split(", ")
            ]
            
            entities.sort(key=lambda x: len(x[0]), reverse=True)
            # Tokenize and reconstruct text
            tokenized_text = tokenizer.tokenize(source_text)
            reconstructed_text = reconstruct_text(tokenized_text)
            print(f"Reconstructed text: {reconstructed_text}")
            # Initialize labels and processed tokens
            labels = ['o'] * len(tokenized_text)
            processed_tokens = [False] * len(tokenized_text)
            print(f"Entities found: {entities}")
            # Find matches and update labels
            for entity, label_prefix in entities:
                matches = find_best_matches_char_basis(entity, reconstructed_text, max_distance_threshold)
                for match, confidence, _, _ in matches:
                    print(f"Trying match '{match}' for '{entity}' with confidence: {confidence}")
                    new_labels = label_matching_sequence(tokenized_text, match, labels.copy(), label_prefix, reconstructed_text)
                    if new_labels != labels:  # Check if new labels were assigned
                        labels = new_labels
                        print(f"Matched '{match}' for '{entity}'")
                        break  # Stop after finding the first match that leads to labeling


            # Append processed data for the current record
            processed_data.append({
                "index": source_index, 
                "tokenized_text": tokenized_text, 
                "labels": labels
            })
            
    return processed_data

# Example usage of the script
max_distance_threshold = 0.6
sourceData = load_json_file('data/parsed_siirtokarjalaiset.json')
apiResults = load_json_file('apiResponse/All_responses10100.json')

processed_data = process_data(sourceData, apiResults, 0, 10000, max_distance_threshold)

save_to_json_file(processed_data, 'nerTaggerTrainData10000withNames.json')
