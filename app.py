from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os 
import streamlit as st

# Load environment variables
load_dotenv()

# Get API key from .env file
api_key = os.getenv("GOOGLE_API_KEY")

# Ensure API key is not missing
if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing! Please add it to your .env file.")

# Initialize Google Generative AI model with API key
model = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key,temperature=1)

st.header("Next word predictor")


user_input = st.text_input("Enter your text")
# Define prompt template
temp = PromptTemplate(
    template="Predict the next  words: {text}",
    input_variables=['text']
)

if st.button("Predict"):
    if user_input.strip():
        chain = temp | model  # Creating a LangChain pipeline
        res = chain.invoke({"text": user_input})  # Correctly passing input variable
        
        if res:
            st.subheader("Predict")
            st.write(res)
        else:
            st.error("Error: No response from the model.")
    else:
        st.warning("Please enter some text to predict words.")
