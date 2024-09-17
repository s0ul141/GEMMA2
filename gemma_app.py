import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


# Cache the model loading to avoid reloading it on every interaction
@st.cache_resource
def load_model():
    model_name = "s0uL141/fine_tuned_science_gemma2b-it"  # Replace with your Hugging Face repo or local path
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type = "nf4", bnb_4bit_compute_dtype = torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config = bnb_congif, device_map={"":0})
    return tokenizer, model
# Load the model and tokenizer
tokenizer, model = load_model()
def generate_response(prompt, max_new_tokens=1000):
    # Tokenize input prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    # Generate response using the model
    output = model.generate(inputs.input_ids, min_length=100, max_new_tokens=max_new_tokens, num_return_sequences=1)
    # Decode the response and return
    return tokenizer.decode(output[0], skip_special_tokens=True)


# Streamlit App
def main():
    # Set up the title and description for the app
    st.title("Fine-Tuned Science Gemma 2b-it Model")
    st.write("This app generates responses based on your input using a fine-tuned version of the Gemma 2b-it model.")

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
