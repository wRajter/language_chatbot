from os import remove
import streamlit as st
from streamlit_chat import message as st_message
from language_chatbot.translate_api import translate
from language_chatbot.detect import detect_language
import requests


#url = "https://chatbot-ni4mcaftla-ew.a.run.app/reply"

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

key_pairs = {'No specific language': 'no lang', 'English': 'en', 'German' : 'de', 'Spanish' : 'es', 'French' : 'fr', 'Italian' : 'it', 'Dutch' : 'nl', 'Polish' : 'pl', 'Portuguese' : 'pt', 'Slovak' : 'sk',
            'Hungarian' : 'hu'}

lang_select = key_pairs[lang_choice]

def generate_answer(url = "http://127.0.0.1:8000/reply"):

    user_message = st.session_state.input_text

    if  lang_select == 'no lang':
        language_detect = detect_language(user_message)
    else:
        language_detect = lang_select

    if len(language_detect) > 8:
        return language_detect

    params = {'text': user_message, "user_language": language_detect}

    response = requests.get(url, params=params)
    answer = response.json()

    session_num = len(st.session_state.history)
    conv = []

    st.session_state.history.append({"message": user_message, "is_user": True, 'key': f'u_{session_num}'})
    st.session_state.history.append({"message": answer['response'], "is_user": False, 'key': f'b_{session_num}'})
    conv.append(translate(user_message, "en"))
    conv.append(answer["response"])

    conv = " ... ".join(conv)

    hist_translated = translate(conv, language_detect).split(' ... ')
    for el in hist_translated:
        if el.startswith(' '):
            el = el[1:]

    if '' in hist_translated:
        hist_translated.remove('')

    bot_answer_tr = hist_translated[-1]

    del st.session_state.history[-1]

    if eng_trans == True:
        if lang_select != "en":
            output = f'''
            {bot_answer_tr}
            ({answer['response']})
            '''

            st.session_state.history.append({"message": output, "is_user": False, 'key': f'b_{session_num}'})

    else:
        st.session_state.history.append({"message": bot_answer_tr, "is_user": False, 'key': f'b_{session_num}'})

st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)

for chat in reversed(st.session_state.history):
    st_message(**chat)
