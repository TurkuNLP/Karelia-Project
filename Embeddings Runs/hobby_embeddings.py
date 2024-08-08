import json
import openai
import os
import csv
import time
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Configure OpenAI API client
openai.api_key = api_key

def extract_data(data, category, limit=None):
    extracted = []
    for key, value in data[category].items():
        extracted.append(key)
        if limit and len(extracted) >= limit:
            break
    return extracted

def get_embeddings(terms, start_index=0):
    embeddings = []
    api_call_count = start_index
    max_retries = 5

    for i, term in enumerate(terms[start_index:], start=start_index):
        success = False
        retries = 0
        while not success and retries < max_retries:
            try:
                response = openai.Embedding.create(input=term, engine="text-embedding-3-small")
                embeddings.append((term, response['data'][0]['embedding']))
                api_call_count += 1
                print(f"API Call {api_call_count}: Embedding fetched for {term}")
                success = True
            except Exception as e:
                retries += 1
                print(f"Error fetching embedding for {term}. Attempt {retries}/{max_retries}. Error: {str(e)}")
                time.sleep(15)
        if not success:
            print(f"Failed to fetch embedding for {term} after {max_retries} attempts. Stopping process at index {api_call_count}.")
            return embeddings, api_call_count

    return embeddings, api_call_count

def save_embeddings_to_csv(embeddings, filename):
    # Check if file exists to write headers only if the file is new
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['term', 'embedding'])  # Write header only if file does not exist
        for term, embedding in embeddings:
            writer.writerow([term, embedding])


def main():
    with open('data/combinedHowManyHobbies2.json', 'r') as file:
        data = json.load(file)

    hobby_limit = 7500
    org_limit = 64000

    hobbies = extract_data(data, 'HobbiesCount', hobby_limit)
    organizations = extract_data(data, 'SocialOrgsCount', org_limit)

    hobby_start_index = 1817  # resume from index after last successful entry
    org_start_index = 67  # example start index for organizations

    hobby_embeddings, last_hobby_index = get_embeddings(hobbies, hobby_start_index)
    org_embeddings, last_org_index = get_embeddings(organizations, org_start_index)

    hobby_filename = 'hobby_embeddings_total.csv'
    org_filename = 'org_embeddings_total.csv'

    save_embeddings_to_csv(hobby_embeddings, hobby_filename)
    save_embeddings_to_csv(org_embeddings, org_filename)

    print(f"Hobby embeddings saved to {hobby_filename} up to index {last_hobby_index}")
    print(f"Organization embeddings saved to {org_filename} up to index {last_org_index}")

if __name__ == "__main__":
    main()

