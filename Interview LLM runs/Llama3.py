


import json
import os
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

# Configure logging and login to Hugging Face
from huggingface_hub import login
login(token="")

def llama3_template(prompt_text_suomi, story):
    user_message = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt_text_suomi.strip()}\n<|eot_id|>"
    story_message = f"<|start_header_id|>user<|end_header_id|>\n\n{story.strip()}\n<|eot_id|>"
    return f"<|begin_of_text|>{user_message}{story_message}<|start_header_id|>assistant<|end_header_id|>"

# Generic function to handle different model templates
def generate_text(story, prompt_text_suomi, pipeline, config, template_function):
    combined_prompt = template_function(prompt_text_suomi, story)

    outputs = pipeline(combined_prompt, 
                       max_new_tokens=config['max_new_tokens'], 
                       num_beams=config['num_beams'], 
                       do_sample=config['do_sample'],
                       temperature=config['temperature'], 
                       top_k=config['top_k'], 
                       top_p=config['top_p'])
    print(outputs)
    return outputs[0]["generated_text"]

def feeder(file_path, prompt_text_suomi, pipeline, config, output_dir, template_function):
    # Load stories from the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        stories = json.load(file)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each story, format it, and generate text
    for i, story in enumerate(stories):
        formatted_story = (
            f"primary_person_name: {story['primary_person_name']}\n"
            f"primary_person_id: {story['primary_person_id']}\n"
            f"spouse_name: {story['spouse_name']}\n"
            f"spouse_id: {story['spouse_id']}\n"
            f"source_text: {story['source_text']}\n"
        )
        
        generated_text = generate_text(formatted_story, prompt_text_suomi, pipeline, config, template_function)
        
        # Print the generated text
        print(f"Generated text for story {i + 1}:\n{generated_text}\n")
        
        # Save the raw response
        with open(f"{output_dir}/batch_{i + 1}.json", 'w', encoding='utf-8') as file:
            json.dump({"response": generated_text}, file, indent=4)
        print(f"Processed and saved batch {i + 1}")

# Example usage
file_path = 'Samples/sample_siirtokarjalaiset_NOT_annotated.json'
base_output_dir = "raw_api_responses_multiple_configs_llama3_SUOMI"

# Define multiple configurations for testing
configs = [
        {"do_sample": True, "num_beams": 1, "temperature": 0.3, "top_k": 50, "top_p": 0.1, "max_new_tokens": 400},
        {"do_sample": False, "num_beams": 1, "temperature": 0.3, "top_k": 50, "top_p": 0.1, "max_new_tokens": 400},
]
prompt_text_suomi = """Tarvitsen sinun kaapivan tiedot tästä tekstistä.
Nämä ovat karjalaisten ihmisten haastatteluja suomeksi kirjoitettuna.
Listaa nimet, ID:t, harrastukset ja sosiaaliset järjestöt.
Huomaa listata puolison harrastukset ja sosiaaliset järjestöt erikseen.
Muista, että aviomies on yleensä ensimmäisenä tarinassa, vaikka hän ei olisi päähenkilö.
Älä listaa ammatteja tai sota-ajan tehtäviä.
Älä ehdota algoritmin tekemistä. Älä listaa lähdetekstiä.
Älä vastaa muuhun kuin pyydettyyn tietoon.
Älä käännä löytöjäsi englanniksi.
Anna vastauksesi perusmuodossa.
Erottele yksi harrastus tai järjestö pilkulla.
Vastaa ainoastaan seuraavassa muodossa jokaiseen tarinaan:
Kun vastaat, anna vastaukset kaikille kategorioille seuraavien avain sanojen jälkeen:

            PersonName: 
            PersonHobbies: 
            PersonSocialOrgs: 
        
            SpouseName:
            SpouseHobbies:
            SpouseSocialOrgs:
            """
# Initialize the prompt text (assuming this is already defined)
prompt_text = """
I need you to scrape data from this text.
This is an interview from Karelian people written in Finnish. 
List me names, hobbies, and social organisations.
Notice to list spouse's hobbies and social orgs separately. 
Do not list jobs, or war time occupations.
Do not answer anything but the asked information. 
List your findings in Finnish. Do not translate your findings. 
Give your answers how they are written in the text. 
Focus on differentiating one hobby or organization with comma. 
Always and strictly respond in the following format for each story:
Give your answers after repeating each category: 

PersonName: 
PersonHobbies: 
PersonSocialOrgs: 

SpouseName:
SpouseHobbies:
SpouseSocialOrgs:
"""

# Initialization and loading model
model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
print(model_id)
shared_dir = "/scratch/project_462000321/joonatan/shared_models"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=shared_dir)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=shared_dir, device_map="auto")

# Initialize Accelerator
accelerator = Accelerator()
model, tokenizer = accelerator.prepare(model, tokenizer)

# Initialize the text generation pipeline
pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

# Run the process with multiple configurations
for config in configs:
    # Define the output directory for the current configuration
    output_dir = f"{base_output_dir}/config_{config['do_sample']}_{config['num_beams']}_{config['temperature']}_{config['top_k']}_{config['top_p']}_{config['max_new_tokens']}"
    
    # Example call to the feeder function
    feeder(file_path, prompt_text_suomi, pipeline, config, output_dir, llama3_template)

