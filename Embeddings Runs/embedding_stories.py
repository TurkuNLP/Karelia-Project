import json
import openai
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import time
import csv

# Load API Key from .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Configure OpenAI API client
openai.api_key = api_key

# Load your dataset
with open('data/parsed_siirtokarjalaiset_with_english.json', 'r') as file:
    data = json.load(file)

# Define the prompt
prompt = ""

# Adjust the source texts by adding the prompt at the beginning
texts = [prompt + " " + item['source_text_english'] for item in data]
index = [item['index'] for item in data]

# Function to get embeddings with retry mechanism
def get_embedding(text, model="text-embedding-3-large", retries=5):
    text = text.replace("\n", " ")  # Replace new lines with spaces in the text
    attempt = 0
    while attempt < retries:
        try:
            response = openai.Embedding.create(input=[text], model=model)
            embedding = response['data'][0]['embedding']
            return embedding
        except Exception as e:
            print(f"An error occurred on attempt {attempt + 1}: {e}")
            time.sleep(15)  # wait for 15 seconds before retrying
            attempt += 1
    print(f"Failed to fetch embedding after {retries} attempts.")
    return None  # Return None if all retries fail

# Starting index for processing
start_index = 0  # Example start index

# Initialize DataFrame if needed
filename = 'All_Stories_english_embeddings.csv'
if not os.path.exists(filename):
    df = pd.DataFrame(columns=['index', 'combined_text', 'embedding'])
    df.to_csv(filename, index=False)

# Process texts and append to CSV
with open(filename, 'a', newline='') as file:
    writer = csv.writer(file)
    if file.tell() == 0:
        writer.writerow(['index', 'combined_text', 'embedding'])

    for i, text in enumerate(texts[start_index:], start=start_index):
        embedding = get_embedding(text)
        if embedding:
            writer.writerow([index[i], text, embedding])
            print(f"Processed {i + 1} out of {len(texts)} and saved to CSV")
        else:
            print(f"Failed to process {i + 1}, stopping script.")
            break
