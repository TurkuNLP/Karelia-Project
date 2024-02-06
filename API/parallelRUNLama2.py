import argparse
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 
from huggingface_hub import login
login(token="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")

def log_gpu_memory():
    for gpu in range(torch.cuda.device_count()):
        allocated_memory = torch.cuda.memory_allocated(gpu) / (1024 ** 3)  # Convert to GB
        max_allocated_memory = torch.cuda.max_memory_allocated(gpu) / (1024 ** 3)
        print(f"GPU {gpu}: Current Memory Allocated: {allocated_memory:.2f} GB, Max Memory Allocated: {max_allocated_memory:.2f} GB")

        
class InferenceModel(pl.LightningModule):
    def __init__(self, model_id, shared_dir, hf_auth):
        super().__init__()
        print("Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=shared_dir)
        print("Tokenizer loaded.")

        print("Loading Model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            use_auth_token=hf_auth,
            cache_dir=shared_dir
        )
        print("Model loaded.")

        
    def generate_text(self, prompt, system_prompt="", max_new_tokens=1024, 
                      temperature=0.6, top_p=0.9, top_k=50, repetition_penalty=1.2):
        # Prepare input data
        conversation = []
        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": prompt})

        input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt")

        # Move input data to the correct device
        input_ids = input_ids.to(self.device)

        # Generate response
        output_ids = self.model.generate(input_ids, 
                                         max_new_tokens=max_new_tokens, 
                                         do_sample=True, 
                                         temperature=temperature, 
                                         top_p=top_p, 
                                         top_k=top_k, 
                                         num_beams=1, 
                                         repetition_penalty=repetition_penalty)

        # Decode and return response
        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return response

def main():
    print(f"Using PyTorch Lightning version: {pl.__version__}")
    
    # Check and print the number of available GPUs
    available_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {available_gpus}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=1, type=int, help='number of GPUs per node')
    args = parser.parse_args()

    # Ensure the requested number of GPUs does not exceed the available GPUs
    if args.gpus > available_gpus:
        print("Warning: Requested more GPUs than are available. Using available GPUs.")
        args.gpus = available_gpus

    # Initialize PyTorch Lightning trainer with the DDP strategy for a single node
    trainer_kwargs = {
        'gpus': args.gpus,
        'accelerator': 'gpu',
        'strategy': 'ddp',
        'precision': 16
    }
    trainer = pl.Trainer(**trainer_kwargs)


    model_id = "meta-llama/Llama-2-70b-chat-hf"
    shared_dir = "/scratch/project_462000321/joonatan/shared_models"
    hf_auth = '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'
    print(model_id)
    # Initialize the model
    model = InferenceModel(model_id, shared_dir, hf_auth)
    log_gpu_memory()

    # Running the generation function on the first GPU only to avoid duplicated outputs
    if trainer.global_rank == 0:
        prompt = "Hello, how are you?"
        response = model.generate_text(prompt)
        print(response)
        
    if trainer.global_rank == 0:
        prompt_prefix = """<<SYS>>
        You act as data scraping bot. 
        Do not generate text, only generate the asked format, and answers for categories. <</SYS>>
            "[INST]"
            I need you to scrape data from this text.
            This is an interview from Karelian people written in finnish. 
            List me names, IDs, hobbies and social organisations.
            Notice to list spouse's hobbies and social orgs separately. 
            Keep in mind that husband is usually listed first in the story. Even when he is not primary person. 
            Do not list jobs, or war time occupations.
            Do not suggest to make an algorithm. Do not list the sourcetext.
            Do not answer anything but the asked information. 
            Do not translate your findings. 
            Give your answers in base form. 
            Differentiate one hobby or organization with comma. 
            Stop once you've gone through the prompted story.
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


        import json
        import time
        import os
        import logging
        import re

        logging.basicConfig(level=logging.INFO)

        # Constants
        batch_size = 1
        SKIP_FIRST_PERSON = False
        output_file_path = 'apiResponse/all_responses_200_sample.json'

        # Load data
        with open('data/sample_siirtokarjalaiset.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        def create_user_message(data_str):
            # Keywords for splitting
            keywords = [
                "index", 
                "primary_person_name", 
                "primary_person_id", 
                "spouse_name", 
                "spouse_id", 
                "source_text"
            ]

            formatted_data_list = []
            for i in range(len(keywords)):
                keyword = keywords[i]
                start_index = data_str.find(keyword)

                # If this isn't the last keyword, find the start of the next keyword for the end index
                end_index = data_str.find(keywords[i+1]) if i+1 < len(keywords) else None

                if start_index != -1:
                    # Extract the key-value pair based on the start and end index
                    pair = data_str[start_index:end_index].strip()
                    formatted_data_list.append(pair)

            formatted_data_str = "\n".join(formatted_data_list)
            return prompt_prefix + formatted_data_str





        def make_api_call(batch_num, combined_message_str):
            combined_message_str = combined_message_str + "[/INST]" # "</s>" 
            print(combined_message_str)

            response = model.generate_text(combined_message_str)

            print("####")

            if response:  # Make sure response is not an empty list
                response_text = response[0]  # Access the first element of the list to get the string
                stripped_response = response_text.replace(combined_message_str, '').strip()
            else:
                stripped_response = ""  # You might want to handle this case differently

            response = {"choices": [{"message": {"content": stripped_response}}]}
            print("####### stripped response", response)

            # Store raw API response
            with open(f"apiResponse/raw_api_responses/batch_{batch_num}.json", 'w', encoding='utf-8') as file:
                json.dump(response, file, ensure_ascii=False)

            return response

        #REGEX and processing functions
        import re

        def extract_person_blocks(response_content):
            person_pattern = re.compile(r'person\s*name', re.IGNORECASE)
            person_starts = [match.start() for match in re.finditer(person_pattern, response_content)]
            person_blocks = []
            for i, start in enumerate(person_starts):
                end = person_starts[i + 1] if i + 1 < len(person_starts) else None
                person_blocks.append(response_content[start:end].strip())
            return person_blocks

        def extract_person_data(person_block, expected_keys):
            extracted_data = {}
            for key in expected_keys:
                # Normalize spaces and make the search case-insensitive
                # Convert camelCase and PascalCase to space-separated words
                spaced_key = re.sub(r'(?<!^)(?=[A-Z])', ' ', key).lower()
                # Create a regex pattern to match variations like "personName" and "Person Name"
                pattern = re.compile(r'(' + re.escape(spaced_key) + r'|' + re.escape(key.lower()) + r')\s*:\s*([^\n]*)', re.IGNORECASE)
                match = pattern.search(person_block)
                if match:
                    extracted_data[key] = match.group(2).strip()
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
                    formatted_key = key.replace(" ", "").capitalize()  # Removes spaces and capitalizes the first letter
                    structured_data.append(f"{formatted_key}: {value}")

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

                combined_message_str = create_user_message(data_str)

                response = make_api_call(batch_num, combined_message_str)

                # Check if response is valid (for simulation purposes, always true)
                if response and 'choices' in response:
                    person_blocks = extract_person_blocks(response['choices'][0]['message']['content'])
                    expected_keys = ["PersonID", "PersonName", "PersonHobbies", "PersonSocialOrgs",
                                     "SpouseID", "SpouseName", "SpouseHobbies", "SpouseSocialOrgs"]
                    batch_responses = prepare_structured_response(batch_num, (batch_num - 1) * batch_size, person_blocks, expected_keys, output_file_path)
                    all_responses.extend(batch_responses)

                # Logging the completion of processing for this batch
                logging.info(f"Finished processing batch {batch_num}. Moving to next batch if available.")
                time.sleep(0.5)
            return all_responses


        def run_all_batches(data, batch_size):
            total_batches = len(data) // batch_size + (1 if len(data) % batch_size else 0)
            return run_batches(range(1, total_batches + 1), data, batch_size, output_file_path)


        # The function run_batches is already defined as you provided.

        # Send just batch number 1:
        #batch_number = [1,2,3]
        #responses = run_batches(batch_number, data, batch_size, output_file_path)

        #!!Running the algorithm !!

        responses = run_all_batches(data, batch_size)

        
    

if __name__ == '__main__':
    main()
