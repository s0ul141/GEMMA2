import streamlit as st
from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer

# Set parameters for model loading
max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection; Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage

# Cache the model loading to avoid reloading it on every interaction
@st.cache_resource
def load_model():
    model_name = "google/gemma-2-2b-it"  # Replace with your Hugging Face repo or local path
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer

# Load the model and tokenizer
model, tokenizer = load_model()

# Enable native 2x faster inference
FastLanguageModel.for_inference(model)

def generate_response(prompt, max_new_tokens=500):
    # Prepare the input prompt
    inputs = tokenizer(
        [
            prompt.format(
                "What is cyber security?",  # input
                "",  # output - leave this blank for generation!
            )
        ], return_tensors="pt").to("cuda")  # Use GPU for faster processing

    # Generate response using the model
    text_streamer = TextStreamer(tokenizer)  # Stream the output as it generates
    output = model.generate(**inputs, streamer=text_streamer, max_new_tokens=max_new_tokens)

    # Decode the generated output and return
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit App
def main():
    # Set up the title and description for the app
    st.title("Fine-Tuned Gemma Model")
    st.write("This app generates responses based on your input using a fine-tuned version of the Gemma model.")

    # Text input area for the user to provide a prompt
    user_input = st.text_area("Enter your prompt here:", height=100)

    # Button to trigger text generation
    if st.button("Generate Response"):
        # Check if user input is provided
        if user_input.strip() == "":
            st.write("Please enter a valid prompt.")
        else:
            with st.spinner("Generating response..."):
                # Generate response using the model
                response = generate_response(user_input)
                # Display the generated response
                st.write("### Model Response:")
                st.write(response)

# Entry point to run the app
if __name__ == "__main__":
    main()
