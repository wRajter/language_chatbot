import streamlit as st
from streamlit_chat import message as st_message

import requests

url = "https://chatbot2-ni4mcaftla-ew.a.run.app/reply"


st.set_page_config(page_title="Multilingual Chatbot", page_icon=":computer:", layout="wide")

#dropdown to select language and reload the page
#pass that into generate answer
#add some text explaining consistency problems
#make the output larger


if "history" not in st.session_state:
    st.session_state.history = []

with st.container():
    st.title("Hi, I am your multilingual chatbot :wave:")
    st.subheader("I am here to help you learn a language of your choice :speech_balloon:")
    st.write("---")



st.subheader(":warning: If you want to change the language please reload the page :warning:")

lang_choice = st.selectbox("What language would you like to choose?", options=['No specific language', 'English', 'German', 'Spanish', 'French', 'Italian', 'Dutch', 'Polish', 'Portuguese', 'Slovak'])

st.write("---")

eng_trans = st.checkbox("Would you like an optional English translation")

def generate_answer(url = "https://chatbot2-ni4mcaftla-ew.a.run.app/reply"):

    key_pairs = {'No specific language': 'no lang', 'English': 'en', 'German' : 'de', 'Spanish' : 'es', 'French' : 'fr', 'Italian' : 'it', 'Dutch' : 'nl', 'Polish' : 'pl', 'Portuguese' : 'pt', 'Slovak' : 'sk',
             'Hungarian' : 'hu'}
    #lang_choice = st.selectbox("What language would you like to choose?", options=['No specific language', 'English', 'German', 'Spanish', 'French', 'Italian', 'Dutch', 'Polish', 'Portuguese', 'Slovak'])
    #lang_choice = st.text_input("Choose a language")
    lang_select = key_pairs[lang_choice]

    user_message = st.session_state.input_text

    if eng_trans == True:
        eng_response = requests.get(url, {"text": user_message, "user_language": "en"})
        eng_answer = eng_response.json()

    params = {"text": user_message, "user_language": lang_select}

    response = requests.get(url, params=params)
    answer = response.json()

    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.history.append({"message": answer['response'], "is_user": False})
    st.session_state.history.append({"message": eng_answer['response'], "is_user": False})

st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)

for chat in st.session_state.history:
    st_message(**chat)
