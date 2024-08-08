import json
import math
from dotenv import load_dotenv
import os
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Ensure raw output directory exists
raw_output_dir = 'data/raw_output'
os.makedirs(raw_output_dir, exist_ok=True)

# Load environment variables from .env file
load_dotenv()

model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
print(model_id)
shared_dir = "/scratch/project_462000642/joonatan/shared_models"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=shared_dir)

# Load the model with Flash Attention 2
with torch.device("cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=shared_dir,
        torch_dtype=torch.float16,
        use_flash_attention_2=True,
        device_map="auto"
    )

# Initialize the text generation pipeline
generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

def generate_text(full_prompt):
    # Configuration settings for different generation scenarios
    configs = [
        {"do_sample": False, "num_beams": 1, "temperature": 0.3, "top_k": 50, "top_p": 0.1, "max_new_tokens": 1200},
        {"do_sample": True, "num_beams": 1, "temperature": 0.3, "top_k": 50, "top_p": 0.1, "max_new_tokens": 1200},
        {"do_sample": True, "num_beams": 1, "temperature": 0.3, "top_k": 50, "top_p": 0.8, "max_new_tokens": 1200},
    ]

    # Select configuration - example using the first configuration
    config = configs[0]

    # Prepare messages in the required format for the model
    messages = [
        {"role": "system", "content": "You are an assistant who categorizes hobbies."},
        {"role": "user", "content": full_prompt}
    ]

    # Generate text using the selected configuration
    outputs = generation_pipeline(messages, 
                                  max_new_tokens=config['max_new_tokens'], 
                                  num_beams=config['num_beams'], 
                                  do_sample=config['do_sample'],
                                  temperature=config['temperature'], 
                                  top_k=config['top_k'], 
                                  top_p=config['top_p'])

    # Assuming the outputs contain the assistant's response, directly returned by the model
    # Adjust according to the model's actual output format.
    assistant_response = outputs[0]["generated_text"][-1]

    return assistant_response

# Example usage
full_prompt = "I enjoy swimming, reading, and playing chess. How would you categorize these hobbies?"
assistant_reply = generate_text(full_prompt)
print(assistant_reply)



import json
import math
from dotenv import load_dotenv
import os
import datetime

# Ensure raw output directory exists
raw_output_dir = 'data/raw_output'
os.makedirs(raw_output_dir, exist_ok=True)

# Load environment variables from .env file
load_dotenv()

# Read the JSON file
with open('data/combinedHowManyHobbies2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
hobbies = list(data['HobbiesCount'].keys())

# Global variables
all_categories = [
    "käsityöt", "yksilöurheilu", "kirjallisuus", "ryhmäurheilu", "musiikki", "valokuvaus",
    "kalastus", "metsästys", "vapaaehtoistoiminta", "pelit", "ei mainintaa", "työ/ammatti", "organisaatiot"
]
categories = {}
unmatched_hobbies = []

def load_data():
    global categories, unmatched_hobbies, all_categories
    
    print("Starting data loading process...")
    
    # Load categories
    try:
        with open('data/categorized_hobbies_flash.json', 'r', encoding='utf-8') as f:
            categories = json.load(f)
        print(f"Successfully loaded categorized hobbies from file.")
        print(f"Number of categories: {len(categories)}")
        
        # Update all_categories with loaded categories
        all_categories = list(set(all_categories + list(categories.keys())))
        
        for category, hobbies in categories.items():
            print(f"  - {category}: {len(hobbies)} hobbies")
        print(f"Total categorized hobbies: {sum(len(hobbies) for hobbies in categories.values())}")
    except FileNotFoundError:
        print("No existing category data found. Initializing with empty categories.")
        categories = {category: [] for category in all_categories}
        print(f"Initialized {len(categories)} empty categories.")
    except json.JSONDecodeError:
        print("Error decoding categorized hobbies JSON. File may be corrupted.")
        categories = {category: [] for category in all_categories}
    
    # Load unmatched hobbies
    try:
        with open('data/unmatched_hobbies_flash.json', 'r', encoding='utf-8') as f:
            unmatched_hobbies = json.load(f)
        print(f"Successfully loaded unmatched hobbies from file.")
        print(f"Number of unmatched hobbies: {len(unmatched_hobbies)}")
        if unmatched_hobbies:
            print("First 5 unmatched hobbies:")
            for hobby in unmatched_hobbies[:5]:
                print(f"  - {hobby}")
            if len(unmatched_hobbies) > 5:
                print(f"  ... and {len(unmatched_hobbies) - 5} more")
    except FileNotFoundError:
        unmatched_hobbies = []
        print("No unmatched hobbies data found. Starting with an empty list.")
    except json.JSONDecodeError:
        print("Error decoding unmatched hobbies JSON. File may be corrupted.")
        unmatched_hobbies = []
    
    print("\nData loading summary:")
    print(f"Total categories: {len(all_categories)}")
    print(f"Categories: {', '.join(all_categories)}")
    print(f"Total categorized hobbies: {sum(len(hobbies) for hobbies in categories.values())}")
    print(f"Total unmatched hobbies: {len(unmatched_hobbies)}")
    print("Data loading process complete.")

def save_data():
    with open('data/categorized_hobbies_flash.json', 'w', encoding='utf-8') as f:
        json.dump(categories, f, ensure_ascii=False, indent=4)
    with open('data/unmatched_hobbies_flash.json', 'w', encoding='utf-8') as f:
        json.dump(unmatched_hobbies, f, ensure_ascii=False, indent=4)
    print("Data saved to files.")

def save_hobby(hobby, category_info):
    global all_categories, categories, unmatched_hobbies
    
    # List of phrases that indicate a new category
    new_category_indicators = ["uusi kategoria,", "new category,"]
    
    # Remove any leading/trailing whitespace
    category_info = category_info.strip()
    
    # Check if this is a new category
    is_new_category = any(indicator in category_info.lower() for indicator in new_category_indicators)
    
    if is_new_category:
        # Extract the new category name
        for indicator in new_category_indicators:
            if indicator in category_info.lower():
                _, category = category_info.lower().split(indicator, 1)
                category = category.strip()
                break
        
        # Create the new category if it doesn't exist
        if category not in categories:
            categories[category] = []
            all_categories.append(category)
    else:
        # Use the provided category directly
        category = category_info
        
        # If the category doesn't exist, add it to unmatched_hobbies
        if category not in categories:
            unmatched_hobbies.append(hobby)
            return None  # Return None to indicate the hobby wasn't categorized
    
    # Add the hobby to the appropriate category
    categories[category].append(hobby)
    
    # Remove the hobby from unmatched_hobbies if it's there
    if hobby in unmatched_hobbies:
        unmatched_hobbies.remove(hobby)
    
    return category  # Return the category name

def destroy_categories(category_names):
    global all_categories  # Ensure we are modifying the global variable
    # Load the current state of data
    load_data()
    
    for category_name in category_names:
        if category_name in categories:
            destroyed_hobbies = categories.pop(category_name)
            unmatched_hobbies.extend(destroyed_hobbies)
            all_categories = [cat for cat in all_categories if cat != category_name]  # Rebuild all_categories
            print(f"Category '{category_name}' has been destroyed. Its hobbies have been moved to unmatched hobbies.")
        else:
            print(f"Category '{category_name}' not found.")
    
    save_data()  # Save data only once after all categories have been processed



def categorize_hobby_batch(hobbies_batch, from_unmatched=False):
    global all_categories 

    introduction = "You are a helpful assistant tasked with categorizing a list of hobbies into given categories."

    # List of hobbies to categorize
    indexed_hobbies = [f"{i+1}. {hobby}" for i, hobby in enumerate(hobbies_batch)]
    hobbies_to_categorize = "Hobbies to categorize:\n" + "\n".join(indexed_hobbies)

    # List of categories
    categories_list = "Available categories:\n" + "\n".join(all_categories)

    special_categories = """
    Special categories for handling noise in the data:
    - ei mainintaa: Use this category only if the entity is blank, says "not mentioned" or "ei mainintaa" or anything similar.
    - työ/ammatti: Use this category only if the entity clearly describes a job or profession and could not be counted as a hobby.
    - organisaatiot: Use this category only if the entity specifically names an organization. For example Marttayhdistys, or Karjalaisseura.
    """

    format_instructions = """
    IMPORTANT: Strictly adhere to the following format in your response:
    index. hobby: category

    If creating a new category, use the format:
    index. hobby: new category, category_name

    Each response MUST be on its own line.
    DO NOT add explanations or extra text to your responses.
    Use only the given categories or create a new category if necessary.

    Examples:
    1. uiminen: yksilöurheilu
    2. lintu bongaus: new category, luonnon havainnointi
    3. ompelu: käsityöt
    """

    new_category_instructions = """
    If a hobby does not fit into any existing category, you can create a new category by following these steps:
    1. Use the exact phrase "new category, " (including the comma and space) followed by the new category name.
    2. The new category name should be general and not too specific.
    3. After creating a new category, use it for the current hobby and any subsequent hobbies that fit into it.
    4. Do not create subcategories or use colons within new category names.
    """

    unmatched_instruction = """\n
    These hobbies are previously unmatched or from too specific categories, 
    please review carefully and be general in your sorting. 
    Note that new categories might have been created where these entities can fit.
    """ if from_unmatched else ""

    full_prompt = (
        introduction + "\n\n" +
        hobbies_to_categorize + "\n\n" +

        special_categories + "\n\n" +
        format_instructions + "\n\n" +
        categories_list + "\n\n" +
        new_category_instructions + "\n\n" +
        unmatched_instruction + "\n\n" +
        "Now categorize the given hobbies:"
    )

    response_text = generate_text(full_prompt)


    # Get the current time
    now = datetime.datetime.now()
    date_time = now.strftime("%d%m%H%M")  # Format as day, month, hour, minute

    # Save the raw response to a file in the raw_output directory
    batch_index = hobbies_batch[0]  # Using the first hobby's index as the file identifier
    raw_output_path = os.path.join(raw_output_dir, f'raw_output_batch_{batch_index}_{date_time}.json')
    with open(raw_output_path, 'w', encoding='utf-8') as f:
        json.dump(response_text, f, ensure_ascii=False, indent=4)
    
    return response_text, hobbies_batch

def process_categorized_data(response, original_hobbies):
    newly_categorized = []
    categorized_data = response.get('content', '')
    lines = categorized_data.split("\n")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        try:
            if ": " in line and ". " in line:
                index_part, category_info = line.split(". ", 1)
                index = int(index_part) - 1
                
                if 0 <= index < len(original_hobbies):
                    original_hobby = original_hobbies[index].strip()
                    category = save_hobby(original_hobby, category_info.split(": ", 1)[1])
                    if category is not None:
                        newly_categorized.append(original_hobby)
                else:
                    print(f"Index out of range: {index}")
            else:
                print(f"Skipping line due to incorrect format: {line}")
        except (ValueError, IndexError) as e:
            print(f"Error processing line: {line}")
            print(f"Exception: {e}")

    save_data()  # Save categorized data after processing the batch
    return newly_categorized


def recategorize_unmatched_hobbies():
    load_data()
    print(f"Loaded {len(unmatched_hobbies)} unmatched hobbies")
    
    # Create a copy of unmatched hobbies to process
    hobbies_to_process = unmatched_hobbies.copy()
    
    batch_size = 50
    num_batches = math.ceil(len(hobbies_to_process) / batch_size)
    
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(hobbies_to_process))
        hobbies_batch = hobbies_to_process[batch_start:batch_end]
        
        print(f"Processing batch {i+1}/{num_batches}")
        
        if not hobbies_batch:
            print("Empty batch encountered. Stopping process.")
            break
        
        try:
            response, original_hobbies = categorize_hobby_batch(hobbies_batch, from_unmatched=True)
            newly_categorized = process_categorized_data(response, original_hobbies)
            
            # Remove categorized hobbies from the unmatched_hobbies list
            unmatched_hobbies[:] = [h for h in unmatched_hobbies if h not in newly_categorized]
            
            # Save updated unmatched_hobbies to JSON after each batch
            save_unmatched_hobbies()
            
            print(f"Batch {i+1} processed. Categorized {len(newly_categorized)} hobbies.")
        except Exception as e:
            print(f"Error processing batch {i+1}: {str(e)}")
        
        print(f"Remaining unmatched hobbies: {len(unmatched_hobbies)}")
    
    print("Recategorization process complete")
    print(f"Final unmatched hobbies count: {len(unmatched_hobbies)}")

def save_unmatched_hobbies():
    with open('data/unmatched_hobbies_flash.json', 'w', encoding='utf-8') as f:
        json.dump(unmatched_hobbies, f, ensure_ascii=False, indent=4)
    print("Unmatched hobbies saved to file.")

def run_categorization():
    load_data()  # Ensure data is loaded at the beginning of the session
    batch_size = 50
    num_batches = math.ceil(len(hobbies) / batch_size)
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(hobbies))
        hobbies_batch = hobbies[batch_start:batch_end]
        print(f"Processing batch {i+1}/{num_batches}")
        response, original_hobbies = categorize_hobby_batch(hobbies_batch, from_unmatched=False)
        process_categorized_data(response, original_hobbies)
        print(f"Batch {i+1} processed")

load_data()



run_categorization()

# destroy_categories(['yleisurheilu','marttatyö'])

categories_to_remove = [
    "ei mainintaa (no clear category, but could be related to work or a specific activity)",
    "kunnossapito: new category, huolto",
    "puhetaidon harjoittaminen could fit into 'puhetaito', but since that category does not exist, it could fit into 'teatteri' or 'viihde', but the best fit is 'puhetaito' which is not available, so 'teatteri' is used: teatteri",
    "mökin rakentaminen could fit into 'rakentaminen', but since that category does not exist for this specific hobby, it could fit into 'kunnossapito' or 'rakentaminen', but the best fit is 'vapaa-ajan asuminen': vapaa-ajan asuminen",
    "histori",
    "maanpuolustus (new category, maanpuolustus)",
    "hevosten hoito (new category, hevosten hoito)",
    "nuorisoseuratoiminnassa -> ei mainintaa (no clear category, but could be related to work or a specific activity), but the best fit is 'yhteiskunnallinen toiminta': yhteiskunnallinen toiminta",
    "golf is not listed, but it could fit into 'pelit' or 'urheilun seuraaminen', but the best fit is 'yksilöurheilu': yksilöurheilu",
    "urheiluvälineiden valmistus, but since that category does not exist for this specific hobby, it could fit into 'käsityöt' or 'urheiluvälineiden valmistus', but the best fit is 'käsityöt': käsityöt",
    "nuorisoseuratoiminnassa -> yhteiskunnallinen toiminta",
    "mökillä viettäminen: new category, mökillä viettäminen is not a good category, so vapaa-ajan asuminen is used: vapaa-ajan asuminen",
    "kellon tekniikka",
    "talviurheilu is already a category, so: talviurheilu",
    "koskenlasku could fit into 'koskenlasku', but since that category does not exist, it could fit into 'koskenlasku' or 'urheilun seuraaminen', but the best fit is 'koskenlasku', so:",
    "nuorisoseuratoiminnassa is already categorized as 'yhteiskunnallinen toiminta', so: yhteiskunnallinen toiminta",
    "käsityökalujen käyttö, ei sopiva, joten: käsityöt",
    "ympäristönsuojelu (new category, ympäristönsuojelu)",
    "kodin ja perheen parissa viettäminen: ei mainintaa, but the best fit is 'kodinhoito': kodinhoito",
    "ruokaan liittyvä harrastus, ei sopiva, joten: käsityöt",
    "ei mainintaa, but the best fit is 'vapaa-ajan asuminen': vapaa-ajan asuminen",
    "rinkbandy",
    "historia",
    "lautapelit",
    "akvaarioharrastus: new category, akvaarioharrastus",
    "akvaarioharrastus",
    "lukeminen ei ole luettelossa, joten: kirjallisuus"
]

#destroy_categories(categories_to_remove)

recategorize_unmatched_hobbies()
