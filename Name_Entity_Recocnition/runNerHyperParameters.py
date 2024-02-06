from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
import torch
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import datasets
import transformers
import evaluate
from pprint import pprint
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import os
import gc
import seqeval

cache_dir='shared_models'
num_labels = 9
tokenizer = AutoTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1", cache_dir=cache_dir)
model = AutoModelForTokenClassification.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1", cache_dir=cache_dir, num_labels=num_labels)
MODEL = "TurkuNLP/bert-base-finnish-cased-v1"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_data(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def process_stories(trimmed_data, tokenizer):
    processed_stories = []

    for story in trimmed_data:
        model_inputs = tokenizer.prepare_for_model(story['input_ids'], 
                                                   truncation=True, 
                                                   padding='max_length', 
                                                   max_length=512)

        processed_story = {
            'input_ids': model_inputs['input_ids'],
            'token_type_ids': model_inputs['token_type_ids'],
            'attention_mask': model_inputs['attention_mask'],
            'labels': story['labels'],
            'ner_tags': story['ner_tags']
        }
        processed_stories.append(processed_story)

    return processed_stories


# Load your trimmed data
trimmed_data_file_path = 'nerTagData_processed.json'
with open(trimmed_data_file_path, 'r', encoding='utf-8') as file:
    trimmed_data = json.load(file)

# Process the data
processed_stories = process_stories(trimmed_data, tokenizer)

# Split the data into train, validation, and test sets
train_data, val_data = train_test_split(processed_stories, test_size=0.1)
#val_data, test_data = train_test_split(temp_data, test_size=(5/15))

# Convert to Hugging Face Dataset objects
train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))
#test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))

# Combine into a single DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
#    'test': test_dataset
})


id2label  = {0: 'o', 1: 'B-P-ORG', 2: 'I-P-ORG', 3: 'B-P-HOB', 4: 'I-P-HOB', 5: 'B-S-ORG', 6: 'I-S-ORG', 7: 'B-S-HOB', 8: 'I-S-HOB'}
label2id ={'o': 0, 'B-P-ORG': 1, 'I-P-ORG': 2, 'B-P-HOB': 3, 'I-P-HOB': 4, 'B-S-ORG': 5, 'I-S-ORG': 6, 'B-S-HOB': 7, 'I-S-HOB': 8} 
label_names = ['o','B-P-ORG', 'I-P-ORG', 'B-P-HOB', 'I-P-HOB', 'B-S-ORG', 'I-S-ORG', 'B-S-HOB', 'I-S-HOB']


# Define a function to align NER tags with tokenized input
def align_ner_tags(example):
    input_ids = example['input_ids']
    ner_tags = example['ner_tags']
    
    # Initialize the list for the aligned NER tags
    aligned_ner_tags = []
    i = 0  # Index for the original ner_tags

    # Align the NER tags with the tokenized input
    for token_id in input_ids:
        token = tokenizer.convert_ids_to_tokens(token_id)
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            aligned_ner_tags.append(-100)  # Special tokens get -100
        else:
            aligned_ner_tags.append(ner_tags[i])
            i += 1  # Move to the next original NER tag

    return {'input_ids': input_ids, 'labels': aligned_ner_tags}

# Apply the function to the entire dataset
processed_dataset = dataset.map(align_ner_tags)

model = transformers.AutoModelForTokenClassification.from_pretrained(
    MODEL,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)




metrics = evaluate.load('seqeval')

def compute_metrics(outputs_and_labels):
    outputs, labels = outputs_and_labels
    predictions = outputs.argmax(axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[id2label[i] for i in label if i != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metrics.compute(predictions=true_predictions, references=true_labels)
    return {
        'precision': all_metrics['overall_precision'],
        'recall': all_metrics['overall_recall'],
        'f1': all_metrics['overall_f1'],
        'accuracy': all_metrics['overall_accuracy'],
    }


def remove_unused_columns(example):
    # Define the keys you want to keep
    keys_to_keep = ['input_ids', 'attention_mask', 'labels']

    
    return {key: example[key] for key in keys_to_keep}

for split in dataset.keys():
    processed_dataset[split] = processed_dataset[split].map(remove_unused_columns, 
                                                  remove_columns=dataset[split].column_names)



from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

from collections import defaultdict

class LogSavingCallback(transformers.TrainerCallback):
    def on_train_begin(self, *args, **kwargs):
        self.logs = defaultdict(list)
        self.training = True

    def on_train_end(self, *args, **kwargs):
        self.training = False

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if self.training:
            for k, v in logs.items():
                if k != "epoch" or v not in self.logs[k]:
                    self.logs[k].append(v)

training_logs = LogSavingCallback()



# Function to clear GPU memory
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Function to create model
def create_model():
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    return model

# Function to create tokenizer
def create_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)

# Function to clear GPU memory and release resources
def release_resources(model, tokenizer):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model
    del tokenizer
    torch.cuda.empty_cache()


def print_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory = torch.cuda.memory_allocated(0)
        used_memory = total_memory - free_memory
        print(f"Total GPU Memory: {total_memory / 1024**3:.2f} GB")
        print(f"Used GPU Memory: {used_memory / 1024**3:.2f} GB")
        print(f"Free GPU Memory: {free_memory / 1024**3:.2f} GB")
    else:
        print("CUDA is not available. No GPU detected.")

warnings.filterwarnings('ignore', message="Some weights of BertForTokenClassification were not initialized from the model checkpoint*.")
warnings.filterwarnings('ignore', message="You're using a BertTokenizerFast tokenizer")
warnings.filterwarnings('ignore', message="Checkpoint destination directory")
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
warnings.filterwarnings('ignore', message=".*o seems not to be NE tag.*")


clear_gpu_memory()
print_gpu_memory()

cache_dir = 'shared_models'
MODEL = "TurkuNLP/bert-base-finnish-cased-v1"
num_labels = 9  

# Define your base training parameters
base_trainer_args = {
    "output_dir": 'checkpoints',
    "evaluation_strategy": 'steps',
    "logging_strategy": 'steps',
    "load_best_model_at_end": True,
    "eval_steps": 100,
    "logging_steps": 100,
    "save_steps": 100,
    "metric_for_best_model": 'f1',
    "save_total_limit": 5,
    "per_device_eval_batch_size": 32
}

# Define ranges for learning rates, batch sizes, and max steps
learning_rates = [0.0001,0.0005,0.00005]
batch_sizes = [64,16,32]
max_steps_options = [1500,2500]
metrics_dict = {}

for lr in learning_rates:
    for batch_size in batch_sizes:
        for max_steps in max_steps_options:
            clear_gpu_memory()
            print_gpu_memory()        
            # Re-create model and tokenizer
            model = create_model()
            tokenizer = create_tokenizer()

            # Update training arguments including max_steps
            trainer_args = TrainingArguments(
                **base_trainer_args, 
                learning_rate=lr,
                per_device_train_batch_size=batch_size,
                max_steps=max_steps,  # Set max_steps here
                report_to=['tensorboard']
            )
            data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

            trainer = Trainer(
                model=model,
                args=trainer_args,
                train_dataset=processed_dataset['train'],
                eval_dataset=processed_dataset['validation'],
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )

            print(f"Training with learning rate: {lr}, batch size: {batch_size}, max steps: {max_steps}")
            trainer.train()


            # Modify the model_save_path to include max_steps
            model_save_path = os.path.join(cache_dir, f'ner-finnish-model-lr-{lr}-bs-{batch_size}-ms-{max_steps}')
            model.save_pretrained(model_save_path)

            training_logs = trainer.state.log_history
            training_loss = [log['loss'] for log in training_logs if 'loss' in log]
            eval_loss = [log['eval_loss'] for log in training_logs if 'eval_loss' in log]
            f1_scores = [log['eval_f1'] for log in training_logs if 'eval_f1' in log]
            epochs = list(range(1, len(training_loss) + 1))

            # Update metrics_key to include max_steps
            metrics_key = f'lr_{lr}-bs_{batch_size}-ms_{max_steps}'
            metrics_dict[metrics_key] = {
                'training_loss': {'epochs': epochs, 'values': training_loss},
                'eval_loss': {'epochs': epochs, 'values': eval_loss},
                'f1_scores': {'epochs': epochs, 'values': f1_scores}
            }


        # Release resources after training each model
        release_resources(model, tokenizer)

        # Clear GPU memory at the end
        clear_gpu_memory()
        print_gpu_memory()

                # Explicitly delete large objects and collect garbage
        del model
        del tokenizer
        del trainer
        del data_collator
        gc.collect()
        torch.cuda.empty_cache()
        
        print_gpu_memory()



def save_metrics_to_txt(metrics_dict, base_file_name, base_path='model_results'):
    for key, value in metrics_dict.items():
        # Create a unique file name for each model configuration
        file_name = f"{base_file_name}_{key}.txt"
        file_path = os.path.join(base_path, file_name)

        with open(file_path, 'w') as file:
            file.write(f"Metrics for {key}:\n")
            for metric_name, metric_data in value.items():
                file.write(f"  {metric_name.title()}:\n")
                for epoch, metric_value in zip(metric_data['epochs'], metric_data['values']):
                    file.write(f"    Epoch {epoch}: {metric_value}\n")
            file.write("\n")

# Example of how to call the function
base_file_name = 'metrics_data'
save_metrics_to_txt(metrics_dict, base_file_name)


# Plotting function
def plot_metrics_for_batch_size(metrics_dict, batch_size, metric_name):
    plt.figure(figsize=(10, 6))
    for lr in learning_rates:
        for max_steps in max_steps_options:
            tag = f'lr_{lr}-bs_{batch_size}-ms_{max_steps}'
            if tag in metrics_dict:
                num_points = len(metrics_dict[tag][metric_name]['values'])
                x_axis = range(1, num_points + 1)  # X-axis representing the number of evaluation/logging steps
                values = metrics_dict[tag][metric_name]['values']
                plt.plot(x_axis, values, label=f'LR: {lr}')
    
    plt.title(f'{metric_name.replace("_", " ").title()} for Batch Size {batch_size}')
    plt.xlabel('Evaluation Step')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.legend()

    

    # Define the file path for saving the plot
    plot_file_name = f"10000_data_{metric_name}_batchsize_{batch_size}.png"
    plot_save_path = os.path.join('model_results', plot_file_name)

    # Save the plot to the file
    plt.savefig(plot_save_path)

    plt.show()
    plt.close()

# Create the model_results directory if it doesn't exist
os.makedirs('model_results', exist_ok=True)

# Iterate over each batch size and plot metrics
for batch_size in batch_sizes:
    for metric_name in ['training_loss', 'eval_loss', 'f1_scores']:
        plot_metrics_for_batch_size(metrics_dict, batch_size, metric_name)



def plot_all_models_combined(metrics_dict, metric_name):
    plt.figure(figsize=(15, 8))  # Adjust the figure size as needed

    for lr in learning_rates:
        for batch_size in batch_sizes:
            for max_steps in max_steps_options:
                tag = f'lr_{lr}-bs_{batch_size}-ms_{max_steps}'
                if tag in metrics_dict:
                    num_points = len(metrics_dict[tag][metric_name]['values'])
                    x_axis = range(1, num_points + 1)  # X-axis representing the number of evaluation/logging steps
                    values = metrics_dict[tag][metric_name]['values']
                    plt.plot(x_axis, values, label=f'LR: {lr}, BS: {batch_size}')

    plt.title(f'Combined {metric_name.replace("_", " ").title()} for All Models')
    plt.xlabel('Evaluation Step')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.legend()



    # Save the plot to the file
    plot_file_name = f"10000_data_combined_{metric_name}.png"
    plot_save_path = os.path.join('model_results', plot_file_name)
    plt.savefig(plot_save_path)

        # Display the plot in the output
    plt.show()
    plt.close()

# Create the model_results directory if it doesn't exist
os.makedirs('model_results', exist_ok=True)

# Plot combined metrics for all models
for metric_name in ['training_loss', 'eval_loss', 'f1_scores']:
    plot_all_models_combined(metrics_dict, metric_name)



