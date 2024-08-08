from transformers import AutoTokenizer
import json

# Model name
model_name = "bigscience/bloom"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the file path of the JSON file
file_path = 'Samples/sample_siirtokarjalaiset_NOT_annotated.json'  # Replace with the path to your JSON file

# Load the JSON data
with open(file_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# Initialize total token count
total_token_count = 0

# Iterate over all documents in the JSON data
for document in json_data:
    # Get source text from document
    text_data = document["source_text"]
    
    # Tokenize the source text
    tokens = tokenizer(text_data)
    
    # Count tokens
    token_count = len(tokens['input_ids'])
    total_token_count += token_count
    
    # Output token count for current document
    print(f"Number of tokens in document {document['index']}: {token_count}")

# Output total token count
print("Total number of tokens:", total_token_count)
