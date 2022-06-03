from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from language_chatbot.gpt_model import chat


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
@app.get("/")
def index():
    return {"greeting": "Hello world"}


@app.get("/reply")

def reply(text, user_language):

    return {'response' : chat(text, user_language)}
