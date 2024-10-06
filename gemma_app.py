import streamlit as st
from unsloth import FastLanguageModel
import torch

def main():
    # Streamlit app title
    st.title("Language Model Inference")

    # User inputs for the model parameters
    max_seq_length = st.number_input("Max Sequence Length:", value=2048, step=1)
    dtype = st.selectbox("Select Data Type:", options=["None", "float16", "bfloat16"], index=0)
    load_in_4bit = st.checkbox("Load in 4-bit?", value=True)

    # Load the model and tokenizer
    if st.button("Load Model"):
        with st.spinner("Loading model..."):
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="google/gemma-2-2b-it",
                max_seq_length=max_seq_length,
                dtype=dtype if dtype != "None" else None,
                load_in_4bit=load_in_4bit,
            )
            st.success("Model loaded successfully!")

    # Input field for the prompt
    alpaca_prompt = st.text_area("Input Prompt:", "What is cyber security?")

    # Generate text based on the input prompt
    if st.button("Generate"):
        if 'model' in locals():  # Check if the model is loaded
            with st.spinner("Generating response..."):
                inputs = tokenizer([alpaca_prompt.format("", "")], return_tensors="pt").to("cuda")
                from transformers import TextStreamer
                text_streamer = TextStreamer(tokenizer)
                generated_output = model.generate(**inputs, streamer=text_streamer, max_new_tokens=500)

                # Display the generated output
                st.text_area("Generated Output:", value=generated_output, height=300)
        else:
            st.warning("Please load the model first.")

# Entry point to run the app
if __name__ == "__main__":
    main()
