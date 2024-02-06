#Imports and API Key and constants
import openai
import json
import time
import os
import logging
import time

logging.basicConfig(level=logging.INFO)
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Configure OpenAI API client
openai.api_key = api_key


# Constants
max_retries = 5
batch_size = 1
SKIP_FIRST_PERSON = False
output_file_path = 'apiResponse/all_responses_200_sample.json'

# Load data and run the main function
with open('data/parsed_siirtokarjalaiset.json', 'r') as f:
    data = json.load(f)

prompt_prefix = """I need you to scrape data from this text. 
            These are interview's from Karelian people written in finnish. 
            I will give you a batch of three stories.
            List me names, IDs, hobbies and social organisations.
            Notice to list spouse's hobbies and social orgs separately. 
            Keep in mind that husband is usually listed first in the story. Even when he is not primary person. 
            Do not list jobs, or war time occupations.
            Do not suggest to make an algorithm. Do not list the sourcetext.
            Do not answer anything but the asked information. 
            Do not translate your findings. 
            Give your answers in base form. 
            Differentiate one hobby or organization with comma. 
            Always response in the following format for each story: 

            --
            
            PersonName: 
            PersonID:
            PersonHobbies: 
            PersonSocialOrgs: 
        
            SpouseName:
            SpouseID:
            SpouseHobbies:
            SpouseSocialOrgs:
            """


prompt_prefix_suomi = """Tarvitsen sinun kaapivan tiedot tästä tekstistä.
Nämä ovat karjalaisten ihmisten haastatteluja suomeksi kirjoitettuna.
Annan sinulle kolmen henkiön tarinat.
Listaa nimet, ID:t, harrastukset ja sosiaaliset järjestöt.
Huomaa listata puolison harrastukset ja sosiaaliset järjestöt erikseen.
Muista, että aviomies on yleensä ensimmäisenä tarinassa, vaikka hän ei olisi päähenkilö.
Älä listaa ammatteja tai sota-ajan tehtäviä.
Älä ehdota algoritmin tekemistä. Älä listaa lähdetekstiä.
Älä vastaa muuhun kuin pyydettyyn tietoon.
Älä käännä löytöjäsi englanniksi.
Anna vastauksesi perusmuodossa.
Erottele yksi harrastus tai järjestö pilkulla.
Vastaa aina seuraavassa muodossa jokaiseen tarinaan:

            PersonName: 
            PersonID:
            PersonHobbies: 
            PersonSocialOrgs: 
        
            SpouseName:
            SpouseID:
            SpouseHobbies:
            SpouseSocialOrgs:
            """


SYSTEM_MESSAGE = {"role": "system", "content": "You are a helpful assistant."}

def create_user_message(data_str):
    return {"role": "user", "content": prompt_prefix_suomi + data_str}


def make_api_call(batch_num, messages):
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=messages,
                temperature=0.8,
                timeout=30
            )
            end_time = time.time()
            print(f"Received API response for batch number {batch_num}. Time taken: {end_time - start_time} seconds")

            raw_response_path = f'apiResponse/raw_api_responses/raw_response_{batch_num}.json'
            with open(raw_response_path, 'w', encoding='utf-8') as raw_file:
                json.dump(response, raw_file, ensure_ascii=False, indent=4)

            return response

        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API error in batch {batch_num}: {str(e)}")
            if "Rate limit exceeded" in str(e):
                logging.info("Rate limit exceeded, waiting for 60 seconds...")
                time.sleep(60)
            else:
                logging.info("Encountered an error, waiting for 10 seconds...")
                time.sleep(10)
            time.sleep(2)
    return None

#REGEX and processing functions
import re

def extract_person_blocks(response_content):
    person_pattern = r'PersonName'
    person_starts = [match.start() for match in re.finditer(person_pattern, response_content)]
    person_blocks = []
    for i, start in enumerate(person_starts):
        end = person_starts[i + 1] if i + 1 < len(person_starts) else None
        person_blocks.append(response_content[start:end].strip())
    return person_blocks

def extract_person_data(person_block, expected_keys):
    extracted_data = {}
    for key in expected_keys:
        pattern = key + r': ([^\n]*)'
        match = re.search(pattern, person_block)
        if match:
            extracted_data[key] = match.group(1).strip()
        else:
            extracted_data[key] = ""  # Assigning an empty string for missing data
    return extracted_data

#Formatting structured response from api data
def prepare_structured_response(batch_num, start_index, person_blocks, expected_keys, output_file_path):
    if SKIP_FIRST_PERSON:
        person_blocks = person_blocks[1:]


    all_responses = []

    for j, person_block in enumerate(person_blocks, start=1):
        person_data = extract_person_data(person_block, expected_keys)

        structured_data = []
        for key, value in person_data.items():
            structured_data.append(f"{key}: {value}")
        api_response_string = "\n".join(structured_data)

        structured_response = {
            "batch_number": batch_num,
            "person_index": start_index + j,
            "api_response": api_response_string
        }

        with open(output_file_path, 'a', encoding='utf-8') as file:
            json.dump(structured_response, file, ensure_ascii=False)
            file.write('\n')

        all_responses.append(structured_response)

    return all_responses


#prepare batches
def prepare_batch_data(batch_num, data, batch_size):
    i = (batch_num - 1) * batch_size  # Start index for this batch

    if len(data) - i < batch_size:
        batch = data[i:]
    else:
        batch = data[i:i + batch_size]

    batch_content = []
    batch_indexes = []

    for person in batch:
        if isinstance(person, dict) and "index" in person:
            batch_content.append('Person : \n' + json.dumps(person))
            batch_indexes.append(str(person["index"]))
        else:
            batch_content.append('Person data missing or malformed : \n' + json.dumps(person))

    return '\n'.join(batch_content), batch_indexes


#MAIN API ALGORITHM

# Ensure directories exist
if not os.path.exists('apiResponse'):
    os.makedirs('apiResponse')
if not os.path.exists('apiRequests'):
    os.makedirs('apiRequests')
if not os.path.exists('apiResponse/raw_api_responses'):
    os.makedirs('apiResponse/raw_api_responses')


def run_batches(batch_numbers, data, batch_size, output_file_path):
    print(f"Entering run_batches with batches: {batch_numbers}")
    all_responses = []

    for batch_num in batch_numbers:
        data_str, batch_indexes = prepare_batch_data(batch_num, data, batch_size)

        messages = [
            SYSTEM_MESSAGE,
            create_user_message(data_str)
        ]

        log_message = f"Making API call for batch number {batch_num} with indexes: {', '.join(batch_indexes)}\nData:\n{data_str}\n"
        with open('apiRequests/api_requests_log.txt', 'a', encoding='utf-8') as log_file:
            log_file.write(log_message)

        response = make_api_call(batch_num, messages)
        if response:
            person_blocks = extract_person_blocks(response['choices'][0]['message']['content'])  
            expected_keys = ["PersonID", "PersonName", "PersonHobbies", "PersonSocialOrgs",
                             "SpouseID", "SpouseName", "SpouseHobbies", "SpouseSocialOrgs"]
            batch_responses = prepare_structured_response(batch_num, (batch_num - 1) * batch_size, person_blocks, expected_keys, output_file_path)
            all_responses.extend(batch_responses)

        time.sleep(0.5)

    return all_responses


def run_all_batches(data, batch_size):
    total_batches = len(data) // batch_size + (1 if len(data) % batch_size else 0)
    return run_batches(range(1, total_batches + 1), data, batch_size, output_file_path)

def run_selected_batches(start_batch, end_batch, data, batch_size, output_file_path):
    # Create a list of batch numbers from start_batch to end_batch
    batch_numbers = list(range(start_batch, end_batch + 1))
    
    # Call the run_batches function with the list of batch numbers
    return run_batches(batch_numbers, data, batch_size, output_file_path)


#!!Running the api algorithm !!

#responses = run_all_batches(data, batch_size)


# Run the batches from 1 to n
responses = run_selected_batches(40001, 60000, data, batch_size, output_file_path)