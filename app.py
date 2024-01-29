from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import streamlit as st

# Instantiate GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Function to generate creative text
def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Streamlit app
st.title("Creative Text Generator")

# User input prompt
user_prompt = st.text_input("Enter your prompt:")

# Generate and display creative text on button click
if st.button("Generate Creative Text"):
    if user_prompt:
        generated_text = generate_text(user_prompt)
        st.text("Generated Text:")
        st.write(generated_text)
    else:
        st.warning("Please enter a prompt.")

# You can run this app using 'streamlit run your_script.py' in the terminal.
