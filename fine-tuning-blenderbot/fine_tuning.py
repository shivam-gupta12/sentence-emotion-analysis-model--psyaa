import torch
import pandas as pd
from transformers import BlenderbotTokenizer, BlenderbotForCausalLM, BlenderbotConfig
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from tqdm import tqdm

tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

model_config = BlenderbotConfig.from_pretrained("facebook/blenderbot-400M-distill")

model = BlenderbotForCausalLM.from_pretrained("facebook/blenderbot-400M-distill", config=model_config)

df = pd.read_csv("/Users/damodargupta/Desktop/sentence-emotion analysis model- psyaa/fine-tuning-blenderbot/train.csv", error_bad_lines=False)

training_text = df['context'] + ' ' + df['prompt'] + ' ' + df['utterances']

input_ids = tokenizer.batch_encode_plus(
    training_text.tolist(),
    padding=True,
    truncation=True,
    max_length=512,  
    return_tensors="pt"
)["input_ids"]

dataset = TextDataset(input_ids)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="/Users/damodargupta/Desktop/sentence-emotion analysis model- psyaa/fine-tuning-blenderbot/fine_tuned_model",
    overwrite_output_dir=True,
    num_train_epochs=10,  
    per_device_train_batch_size=4,  
    save_steps=500, 
    save_total_limit=2, 
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

with tqdm(total=training_args.num_train_epochs, desc="Training") as pbar:
    def update_pbar_callback(eng):
        pbar.update()
    trainer.add_event_handler(Trainer.EPOCH_COMPLETED, update_pbar_callback)
    trainer.train()

trainer.save_model("/Users/damodargupta/Desktop/sentence-emotion analysis model- psyaa/fine-tuning-blenderbot/fine_tuned_model")