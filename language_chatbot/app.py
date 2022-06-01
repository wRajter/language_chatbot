import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

st.set_page_config(page_title="Multilingual Chatbot", page_icon=":computer:", layout="wide")

with st.container():
    st.title("Hi, I am your multilingual chatbot :wave:")
    st.subheader("I am here to help you learn a language of your choice :speech_balloon:")
    st.write("---")
