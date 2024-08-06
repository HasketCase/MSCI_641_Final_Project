import wandb
import yaml
from transformers import TrainingArguments, Trainer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import Dataset
import pandas as pd
import json
import torch
from sklearn.metrics import accuracy_score

# This is creating the numeric equivalents to the three different labels. 
label_mapping = {
    "passage": 0,
    "phrase": 1,
    "multi": 2
}

# Initialize empty lists to store paragraphs and tags. The label is paragraphs from the original iterations when the article text was used, the names haven't been updated
paragraphs = []
tags = []

# Didnt define a function, because only the two json files are being loaded this time around. It just snags the title of the article and its tag
with open('train.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line.strip())
        paragraph = data['targetTitle']
        tag = data['tags']
        if isinstance(tag, list):# This allows us to convert the tags to integers, because currently our tags are in a list which is not hashable.
            tag = tag[0]
        paragraphs.append(paragraph)
        tags.append(tag)
# same stuff as abovee=
paragraphs2 = []
tags2 = []
with open('validation.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line.strip())
        paragraph2 = data['targetTitle']
        tag2 = data['tags']
        if isinstance(tag2, list): 
            tag2 = tag2[0]
        paragraphs2.append(paragraph2)
        tags2.append(tag2)

# Convert tags to integers using the mapping
tags = [label_mapping[tag] for tag in tags]
tags2 = [label_mapping[tag] for tag in tags2]

# A function was defined because a sweep was defined to test different combinations of hyperparameters
# The sweep that is called in the bottom uses the yaml file to go through the iterations
# Because more than 30 iterations were performed, wandb was used to track the different outcomes of the tunning process
# Code ran overnight, so I could not always track the results.
def train():
    # Initialize WandB without specifying the config
    wandb.init(project="Task1Tunning")
    config = wandb.config
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased') #Loads the tokenizer required specifically for distilBert
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3) #Grabbing the pre-trained model

    # putting the train and validation into pandas dataframes
    df_train = pd.DataFrame({
        'paragraphs': paragraphs,
        'label': tags
    })

    df_val = pd.DataFrame({
        'paragraphs': paragraphs2,
        'label': tags2
    })

    train_dataset = Dataset.from_pandas(df_train)
    val_dataset = Dataset.from_pandas(df_val)
    # This is the function that tokenizes the text in the article titles. It is used for both train and validation 
    def tokenize_function(examples):
        return tokenizer(examples['paragraphs'], padding='max_length', truncation=True, max_length=512)

    #has the text tokenized
    train_dataset = train_dataset.map(tokenize_function, batched=True) 
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # The hyperparameters. None of them are defined because of the sweep. It grabs the configuration of the hyperparameters from the yaml file.
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        warmup_steps=config.warmup_steps,
        learning_rate=wandb.config.learning_rate,
        weight_decay=config.weight_decay,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        report_to="wandb"
    )
    # Prepares the trainer before it gets called. 
    trainer = Trainer(
        model=model,
        args=training_args, #collects the arguments outlined above
        train_dataset=train_dataset, #grabs the processed and tokenized train data.
        eval_dataset=val_dataset, #grabs the processed and tokenized validation data
        compute_metrics=lambda p: {
            'accuracy': accuracy_score(p.label_ids, p.predictions.argmax(-1)) #Although the eval_accuracy gets printed, this just sets up the validation accuracy to print at the end of the training session
        }
    )

    # Calling the function to train the model! We then produce the evaluation results built into the trainer.
    trainer.train()
    eval_results = trainer.evaluate()
    accuracy = eval_results['eval_accuracy']
    print(f"Val Acc: {accuracy:.4f}")

    # This will take evaluation metric of accuracy and uploads it online to wandb.
    wandb.log({'eval_accuracy': accuracy})

    # If the model has poor performance, I did not care to test it on Kaggle. So this tracks the best accuracy from the sweep, so midway through the running the code, I can grab the most
    # updated accuracy. 
    global_best_accuracy = 0.0
    best_model_path = './best_model'

    # Saving the model so that it can loaded in a separate script. Because the nature of the sweeps, and the runtime of 20 iterations would take more than 20 hours.  
    if accuracy > global_best_accuracy:
        global_best_accuracy = accuracy
        model.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)
    #this is the point which is at the end of the iteration.
    wandb.finish()

# The yaml file has selected sets of hyperparameters to be tested. 
with open('sweep_config2.yaml', 'r') as file:
    sweep_config = yaml.safe_load(file)

# This is what calls the function. Count defines how many runs it will perform. The number of hyperparameters was subject to change. 
sweep_id = wandb.sweep(sweep=sweep_config, project="Task1Tunning")
wandb.agent(sweep_id, function=train, count=16)