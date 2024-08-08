import json
import os
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

# Configure logging and login to Hugging Face
from huggingface_hub import login
login(token="")

def generate_text(story, prompt_text, pipeline, config):
    messages = [
        {"role": "user", "content": prompt_text + story}
    ]

    # Use the apply_chat_template method from the tokenizer
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    print("Raw input to the model:\n", prompt)

    outputs = pipeline(prompt, 
                       max_new_tokens=config['max_new_tokens'], 
                       num_beams=config['num_beams'], 
                       do_sample=config['do_sample'],
                       temperature=config['temperature'], 
                       top_k=config['top_k'], 
                       top_p=config['top_p'])
    
    print("Raw output from the model:\n", outputs)
    return outputs[0]["generated_text"]

def main():
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer_id = "mistralai/Mixtral-8x7B-v0.1"
    shared_dir = "/scratch/project_462000321/joonatan/shared_models"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, cache_dir=shared_dir)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=shared_dir, device_map="auto")

    # Initialize Accelerator
    accelerator = Accelerator()
    model, tokenizer = accelerator.prepare(model, tokenizer)

    # Initialize the text generation pipeline
    pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

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
        Stop answer immediately once you've gone through the prompted story.
        Always respond in the following format for each story: 

        PersonName: 
        PersonHobbies: 
        PersonSocialOrgs: 

        SpouseName:
        SpouseHobbies:
        SpouseSocialOrgs:
    """

    # Define a list of hyperparameter configurations to test
    configs = [
        {"do_sample": True, "num_beams": 1, "temperature": 0.3, "top_k": 50, "top_p": 0.8, "max_new_tokens": 400},

    ]

    # Load stories from a JSON file
    stories = json.load(open('Samples/sample_siirtokarjalaiset_NOT_annotated.json', 'r', encoding='utf-8'))

    # Ensure the output directory exists
    os.makedirs("raw_api_responses6", exist_ok=True)

    for config in configs:
        run_id = f"config_{config['do_sample']}_{config['num_beams']}_{config['temperature']}_{config['top_k']}_{config['top_p']}_{config['max_new_tokens']}"
        os.makedirs(f"raw_api_responses6/{run_id}", exist_ok=True)
        for i, story in enumerate(stories):
            # Extract required fields and create plain text story
            plain_text_story = f"Primary Person: {story['primary_person_name']}\nSpouse: {story['spouse_name']}\nStory: {story['source_text']}"
            
            generated_text = generate_text(plain_text_story, prompt_text, pipeline, config)
            
            # Save the raw response
            with open(f"raw_api_responses6/{run_id}/batch_{i + 1}.json", 'w', encoding='utf-8') as file:
                json.dump({"response": generated_text}, file, indent=4)
            print(f"Processed and saved batch {i + 1} for {run_id}")

if __name__ == "__main__":
    main()




