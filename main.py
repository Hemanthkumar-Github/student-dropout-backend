from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from agent import get_next_question, sessions
from model import predict_dropout
from database import cursor, conn



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/chat")
def chat(session_id: str, msg: str = None):
    question = get_next_question(session_id, msg)

    if question == "DONE":
        data = sessions[session_id]["data"]
        prediction, prob = predict_dropout(data)

        return {
            "prediction": prediction,
            "dropout_probability": f"{prob}%"
        }

    return {"question": question}
