import streamlit as st
from streamlit_chat import message as st_message

import requests

url = "http://127.0.0.1:8000/reply"

st.set_page_config(page_title="Multilingual Chatbot", page_icon=":computer:", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

with st.container():
    st.title("Hi, I am your multilingual chatbot :wave:")
    st.subheader("I am here to help you learn a language of your choice :speech_balloon:")
    st.write("---")

def generate_answer(url = "http://127.0.0.1:8000/reply"):
    user_message = st.session_state.input_text
    response = requests.get(url, params=user_message)
    answer = response.json()

    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.history.append({"message": answer, "is_user": False})

st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)

for chat in st.session_state.history:
    st_message(**chat)
