import streamlit as st
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling,
)
import torch
import os

st.title("GPT-2 Fine-tune & Generate Quotes")

# Section 1: Upload or edit dataset
st.header("Step 1: Prepare Dataset")

uploaded_file = st.file_uploader("Upload your quotes text file (.txt)", type=["txt"])

if uploaded_file:
    dataset_text = uploaded_file.read().decode("utf-8")
else:
    # Default sample quotes
    dataset_text = """The only limit to our realization of tomorrow is our doubts of today.
Do not watch the clock. Do what it does. Keep going.
Success is not final, failure is not fatal: It is the courage to continue that counts.
Believe you can and you're halfway there.
Act as if what you do makes a difference. It does.
"""

dataset_text = st.text_area("Edit your quotes dataset below:", dataset_text, height=200)

dataset_path = "quotes_streamlit.txt"
with open(dataset_path, "w", encoding="utf-8") as f:
    f.write(dataset_text)

# Section 2: Fine-tune button
st.header("Step 2: Fine-tune GPT-2")

if st.button("Start Fine-tuning"):
    with st.spinner("Fine-tuning GPT-2... this may take a few minutes"):
        # Load tokenizer and model
        model_name = "gpt2"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)

        # Prepare dataset for Trainer
        def load_dataset(file_path, tokenizer, block_size=128):
            return TextDataset(
                tokenizer=tokenizer,
                file_path=file_path,
                block_size=block_size,
                overwrite_cache=True,
            )

        train_dataset = load_dataset(dataset_path, tokenizer)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir="./gpt2-finetuned-streamlit",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            logging_steps=10,
            logging_dir="./logs",
            report_to=[],  # disables wandb or other trackers
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        trainer.train()
        trainer.save_model("./gpt2-finetuned-streamlit")
        tokenizer.save_pretrained("./gpt2-finetuned-streamlit")

        st.success("Fine-tuning complete! Model saved.")

# Section 3: Generate quotes
st.header("Step 3: Generate Quotes")

prompt = st.text_input("Enter prompt to generate quote:", "Success is")

if st.button("Generate"):
    if not os.path.exists("./gpt2-finetuned-streamlit"):
        st.error("Please fine-tune the model first!")
    else:
        tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned-streamlit")
        model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned-streamlit")

        inputs = tokenizer(prompt, return_tensors="pt")

        outputs = model.generate(
            **inputs,
            max_length=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=3,
            temperature=0.7,
        )

        st.write("### Generated Quotes:")
        for i, output in enumerate(outputs):
            text = tokenizer.decode(output, skip_special_tokens=True)
            st.write(f"{i+1}. {text}")

