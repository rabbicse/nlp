from fastapi import FastAPI
from starlette import status

from services import chatbot_service

app = FastAPI(debug=True)


@app.post('/api/chatbot', status_code=status.HTTP_201_CREATED)
def ask_question(question: str):
    return chatbot_service.ask_question(question)
