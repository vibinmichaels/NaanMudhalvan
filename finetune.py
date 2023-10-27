from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
import pickle


model_name = "gpt2"  
config = GPT2Config.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, config=config)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)


training_args = TrainingArguments(
    output_dir="./fitnessbot",  
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=1000,
    save_total_limit=2,
)


dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="fitness_data.txt",  
    block_size=256, 
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)


trainer.train()


model_to_save = trainer.model
with open("fine_tuned_model.pkl", 'wb') as model_file:
    pickle.dump(model_to_save, model_file)


print("Welcome to the Fitness Chatbot. Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")

    if user_input.lower() == 'exit':
        print("Fitness Chatbot: Goodbye!")
        break


    input_ids = tokenizer.encode("You: " + user_input, return_tensors="pt").to(device)
    bot_input_ids = input_ids


    response = model.generate(bot_input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    bot_response = tokenizer.decode(response[0], skip_special_tokens=True)

    print("Fitness Chatbot:", bot_response)
