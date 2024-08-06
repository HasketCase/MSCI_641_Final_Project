import json
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import Dataset
import torch

# Using the same mapping from the train process
label_mapping = {
    "passage": 0,
    "phrase": 1,
    "multi": 2
}

paragraphs = []
# Grabs the data the same way it did from the other script
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            paragraph = data['targetTitle']
            paragraphs.append(paragraph)
    return paragraphs

# Snag the data using the function
test_paragraphs = load_data('test.jsonl')

# Load the trained model and tokenizer
model_save_path = './best_model'
tokenizer = DistilBertTokenizer.from_pretrained(model_save_path)
model = DistilBertForSequenceClassification.from_pretrained(model_save_path)

# Create test dataset
df_test = pd.DataFrame({
    'paragraphs': test_paragraphs
})

test_dataset = Dataset.from_pandas(df_test)
# The proper tokenization technique
def tokenize_function(examples):
    return tokenizer(examples['paragraphs'], padding='max_length', truncation=True, max_length=512)


# Formats the data so that it can be feed into the model accordingly
test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# The function that will make the predictions. It will do this in batches of 8. 
def predict(test_dataset):
    predictions = [] #creating a list for the predictions to be stored in
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8)
    model.eval() # setting the model to be in evaluation mode
    with torch.no_grad(): # Making sure no gradients are calculated throughout the process
        for batch in dataloader: #Iterates through each batch of 8
            inputs = {k: v.to(model.device) for k, v in batch.items()} #takes in the inputs
            outputs = model(**inputs) #using these inputs and running them through the model. 
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1) #the logits are now converted to values, and the highest predicted value is used. 
            predictions.extend(preds.cpu().numpy()) #adds the predictions to the predictions list
    return predictions

#We get to call our function and see if it even worked, and if so, if we have any meaningful results
test_predictions = predict(test_dataset)


# Makes a simple dictionary that makes the reverse of the label mapping for the last step.
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Convert predictions to labels
predicted_labels = [reverse_label_mapping.get(pred, 'unknown') for pred in test_predictions]

# We save the datafram as a csv so that we can submit on kaggle
df_results = pd.DataFrame({
    'paragraphs': test_paragraphs,
    'predicted_labels': predicted_labels
})

df_results.to_csv('test_predictions2.csv', index=False)