from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Payload(BaseModel):
    message: str
    user:    str = ""

@app.post("/agent")
async def agent(payload: Payload):
    print(f"\n📩 Message recu: {payload.message}")
    print(f"   De: {payload.user}")
    # Reponse temporaire — l'agent LangGraph viendra ici
    return {"reply": f"✅ Message recu: {payload.message}"}
