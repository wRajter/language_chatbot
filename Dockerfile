FROM python:3.8.13-buster

COPY api /api
COPY language_chatbot /language_chatbot
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt

CMD uvicorn language_chatbot.api:app --host 0.0.0.0 --port $PORT
