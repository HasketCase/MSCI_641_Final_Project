import jsonlines
import pandas as pd
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch

# As we require to load three different datasets, a function is created to read the jsonl file, and return each object as an element of a list.
def extract_from_jsonl_file(file_path):
    with jsonlines.open(file_path) as reader:
        return [clickbait for clickbait in reader]

# We define three separate lists that will take in all the information for each clickbait
full_train = extract_from_jsonl_file('train.jsonl')

print(full_train[0])
print(full_train[1000])
full_val = extract_from_jsonl_file('validation.jsonl')
full_test = extract_from_jsonl_file('test.jsonl')

# Now we define a proccess that pulls the desired input and output properties. Essentially it combines the target paragraphs and the spoilers for each clickbait post. 
# It essentially sets up the keys and structure for the dictionary
def formatted_input_and_output(data, include_target=True):
    formatted_data = []
    for junk in data:
        inputs = " ".join(junk.get("targetParagraphs", []))
        if include_target:
            spoilers = " ".join(junk.get("spoiler", [])) #Does this step in the if statement, so that it doesn't apply to the test set.
            formatted_data.append({"input_text": inputs, "target_text": spoilers})
        #Because our test set does not contain the spoiler property, we choose to only format the target paragraphs
        else:
            formatted_data.append({"input_text": inputs})
    return formatted_data

train_formatted = formatted_input_and_output(full_train)

print(train_formatted[0])
print(train_formatted[1])



val_formatted = formatted_input_and_output(full_val)
test_formatted = formatted_input_and_output(full_test)

# It was recommended that the pandas dataframes be used, as T5 better processes tabular formats. This process is repeated for the test set when necessary
train_dataset = Dataset.from_pandas(pd.DataFrame(train_formatted))

print(train_dataset[1])
val_dataset = Dataset.from_pandas(pd.DataFrame(val_formatted))

model = T5ForConditionalGeneration.from_pretrained("t5-base") # Grabbing the T5 model and assigning it to "model"
t5_text_tokenizer = T5Tokenizer.from_pretrained("t5-base") #loading the respective tokenizer, so that the data can be processed properly

# Because our data has been structured as a dictionaries as setup from pandas and the formatting stage, we can now feed it into here. 
def preprocess_function(examples):
    article_content = examples["input_text"]
    tokenized_content = t5_text_tokenizer(article_content, max_length=512, truncation=True)
    if "target_text" in examples: # We must have the if statement, as our test set will not have the target text in the dictionary, because it is not present in the test.jsonl file
        targets = examples["target_text"]
        with t5_text_tokenizer.as_target_tokenizer():
            phrase_passage_multi = t5_text_tokenizer(targets, max_length=128, truncation=True)
        tokenized_content["labels"] = phrase_passage_multi["input_ids"]
        return tokenized_content
    else:
        tokenized_content = t5_text_tokenizer(article_content, max_length=512, truncation=True)
        return tokenized_content
    
# We call the functions that will tokenize our text. 
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# This function created by Hugging Face will batch and pad our text so that it is prepared for Seq2Seq handleing. 
data_batched_and_paddedlator = DataCollatorForSeq2Seq(t5_text_tokenizer, model=model)

# Listing the hyperparameters for the model. No sweep was included for this model, so tunning took place here. Weight_decay was included. fp16 was attempted to expedite the process
# Batch sizes were increased to 16 to save time. Epochs changed from one run of the model to the next.
hyperparameters = TrainingArguments(
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1e-5,
    output_dir="./results",
    eval_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_steps=10,
    num_train_epochs=4,
    fp16=True,
    weight_decay=0.1
)

# Prepares the trainer before it gets called. 
trainer = Trainer(
    model=model,
    args=hyperparameters, #collects the arguments outlined above
    data_collator=data_batched_and_paddedlator, #Calls the hugging face's collating process outlined above
    train_dataset=train_dataset, #grabs the processed and tokenized train data.
    eval_dataset=val_dataset, #grabs the processed and tokenized validation data
)

# Grabbing the test data, and tokenizes it like it was done before. 
test_inputs = t5_text_tokenizer([example["input_text"] for example in test_formatted], return_tensors="pt", padding=True, truncation=True, max_length=512)
input_ids = test_inputs["input_ids"]

# Calling the function to train the model! We then produce the evaluation results built into the trainer
trainer.train()
eval_results = trainer.evaluate()
print("results", eval_results)

# Now that we are changing course and using our recently built model on our test data to make prediction, we must change the model to evaluation mode
model.eval()

# We now take the input ids of all the tokenized test data. This is the function that will generate the spoilers. 
with torch.no_grad():
    coded_outputs = model.generate(input_ids, max_length=128)

# We take the outputs from the run, and decode them so that we can visually infer the results. 
predicted_spoilers = t5_text_tokenizer.batch_decode(coded_outputs, skip_special_tokens=True)

# We only really need the second columns, but we match it up with the first column to ensure nothing went astray during the process
results_df = pd.DataFrame({
    "input_text": [x["input_text"] for x in test_formatted],
    "predicted_spoiler": predicted_spoilers
})

# We save the new predictions to a csv.
results_df.to_csv('predictions3.csv', index=False)

# This next section of code was done on a separate script, the reason it was done was because some of the text used was only in utf-8 and not what would satisfy the submission on kaggle. 
# It got cumbersome to manually find all the issues 

df = pd.read_csv('predictions3.csv')
second_column = df.iloc[:, 1]
second_column.to_csv('output3.csv', index=False, header=False)

#These were the four characters found in the generated spoilers that didn't comply with what the kaggle submission accecpted. Its probably possible that some other runs may include
#more of these characters. 
def replace_characters(text):
    if pd.isna(text):
        return text
    text = text.replace('’', "'") 
    text = text.replace('—', '-')  
    text = text.replace('é', 'e')  
    text = text.replace('–', '-')
    return text

def process_csv(rough_spoilers, clean_submittable_spoilers):
    df = pd.read_csv(rough_spoilers)
    for col in df.columns:
        df[col] = df[col].apply(replace_characters)
    df.to_csv(clean_submittable_spoilers, index=False)

rough_spoilers = 'output3.csv'  
clean_submittable_spoilers = 'Task2Submission3.csv'
process_csv(rough_spoilers, clean_submittable_spoilers)