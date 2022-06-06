FROM python:3.8.13-buster

COPY language_chatbot /language_chatbot
COPY requirements.txt /requirements.txt
COPY .env /.env

RUN pip install -r requirements.txt

CMD uvicorn language_chatbot.api:app --host 0.0.0.0 --port $PORT
