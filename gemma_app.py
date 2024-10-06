import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Title and Description for the App
st.title("Cybersecurity Fine-Tuned Model Inference")
st.write("This app uses a fine-tuned 'Gemma-2-2b-it' model for cybersecurity-related text generation.")

# Load Model and Tokenizer from Hugging Face
@st.cache_resource
def load_model():
    model_name = "s0uL141/Cyber_gemma2_2B_it"  # Your fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    return model, tokenizer

# Load the model and tokenizer
model, tokenizer = load_model()

# Function to generate text
def generate_text(prompt, max_length=500):
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs.input_ids, 
        max_length=max_length, 
        num_return_sequences=1, 
        do_sample=True,
        temperature=0.7,  # Control the creativity
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Main function to run the app
def main():
    # Create a text input for the user to enter a prompt
    st.write("Enter your cybersecurity-related prompt below:")
    user_prompt = st.text_area("Prompt", "What is cyber security?", height=100)

    # When the user clicks the button, generate the response
    if st.button("Generate Response"):
        if user_prompt.strip() != "":
            with st.spinner("Generating response..."):
                response = generate_text(user_prompt)
                st.write("### Generated Response:")
                st.write(response)
        else:
            st.write("Please enter a valid prompt.")

    # Additional option to control the max length of generated text
    max_length = st.slider("Max Length of Response", min_value=50, max_value=1000, value=500)

    # Footer
    st.write("Fine-tuned model hosted on Hugging Face: [Cyber_gemma2_2B_it](https://huggingface.co/s0uL141/Cyber_gemma2_2B_it)")

# Ensure the script runs directly
if __name__ == "__main__":
    main()
