import json
import os
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

# Configure logging and login to Hugging Face
from huggingface_hub import login
login(token="")

def apply_chat_template(tokenizer, prompt_text, story):
    messages = [{"role": "user", "content": prompt_text + story}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

# Generic function to handle different model templates
def generate_text(story, prompt_text, pipeline, config, tokenizer):
    combined_prompt = apply_chat_template(tokenizer, prompt_text, story)

    outputs = pipeline(combined_prompt, 
                       max_new_tokens=config['max_new_tokens'], 
                       num_beams=config['num_beams'], 
                       do_sample=config['do_sample'],
                       temperature=config['temperature'], 
                       top_k=config['top_k'], 
                       top_p=config['top_p'])
    print(outputs)
    return outputs[0]["generated_text"]

def feeder(file_path, prompt_text, pipeline, config, output_dir, tokenizer):
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
        
        generated_text = generate_text(formatted_story, prompt_text, pipeline, config, tokenizer)
        
        # Print the generated text
        print(f"Generated text for story {i + 1}:\n{generated_text}\n")
        
        # Save the raw response
        with open(f"{output_dir}/batch_{i + 1}.json", 'w', encoding='utf-8') as file:
            json.dump({"response": generated_text}, file, indent=4)
        print(f"Processed and saved batch {i + 1}")

# Example usage
#Samples/sample2/sample2_siirtokarjalaiset_NOT_annotated.json
#file_path = 'Samples/sample_siirtokarjalaiset_NOT_annotated.json'
file_path = 'Samples/sample2/sample2_siirtokarjalaiset_NOT_annotated.json'
base_output_dir = "raw_api_responses_multiple_configs_Qwen2_test"

# Define multiple configurations for testing
configs = [
    {"do_sample": False, "num_beams": 1, "temperature": 0.3, "top_k": 50, "top_p": 0.1, "max_new_tokens": 400},
    {"do_sample": True, "num_beams": 1, "temperature": 0.3, "top_k": 50, "top_p": 0.1, "max_new_tokens": 400},
    {"do_sample": True, "num_beams": 1, "temperature": 0.3, "top_k": 50, "top_p": 0.8, "max_new_tokens": 400},
]

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
Stop answering immediately once you've gone through the prompted story.
Always respond in the following format for each story: 

PersonName: 
PersonHobbies: 
PersonSocialOrgs: 

SpouseName:
SpouseHobbies:
SpouseSocialOrgs:
"""

# Initialization and loading model
model_id = "Qwen/Qwen2-72B-Instruct"
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
    feeder(file_path, prompt_text, pipeline, config, output_dir, tokenizer)
