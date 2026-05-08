from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from tools import (save_trip, set_price, get_report, get_profit,
                   list_entreprises, delete_trip,
                   get_report_pdf, get_profit_pdf, get_summary_pdf)
from dotenv import load_dotenv
from datetime import date
from typing import Optional
import os

load_dotenv()

app = FastAPI()

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "gpt-4o"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)

tools  = [save_trip, set_price, get_report, get_profit,
          list_entreprises, delete_trip,
          get_report_pdf, get_profit_pdf, get_summary_pdf]
memory = MemorySaver()
graph  = create_react_agent(llm, tools, checkpointer=memory)

SYSTEM_PROMPT = f"""Tu es un assistant de gestion de transport de marchandises.
La date du jour est : {date.today().strftime('%d/%m/%Y')}.
Le mois actuel est : {date.today().month}, l annee actuelle est : {date.today().year}.

Tu aides a:
- Lire les recus image et extraire: numero camion, tonnage, entreprise
- Le tonnage dans les recus est toujours en KG — toujours passer unite="kg" a save_trip
- Enregistrer les voyages avec les prix fournis par l utilisateur
- Mettre a jour les prix d un voyage existant
- Supprimer un voyage par son ID
- Generer les rapports et benefices par entreprise (texte ou PDF)
- Generer un PDF global toutes entreprises avec get_summary_pdf

Regles importantes:
- Reponds toujours dans la meme langue que l utilisateur
- Sans precision de mois, utiliser le mois et l annee actuels
- Quand le nom d entreprise n existe pas exactement, choisir automatiquement le plus proche
- Pour supprimer: appelle get_report pour trouver l ID puis delete_trip
- Quand tu recois une image de recu ET des prix, enregistre immediatement
- Prix camion = ce que tu paies au transporteur par tonne
- Prix client = ce que l entreprise te paie par tonne

Regles PDF IMPORTANTES:
- PDF une entreprise rapport   → get_report_pdf
- PDF une entreprise benefice  → get_profit_pdf
- PDF toutes entreprises / resume global / synthese / خلاصة / ملخص → get_summary_pdf (UN SEUL APPEL)
- Ne jamais appeler plusieurs tools PDF en meme temps
- Quand un tool retourne PDF_PATH:..., reponds UNIQUEMENT: PDF_READY:<chemin>
- Sois concis et utilise les emojis pour WhatsApp
"""

class Payload(BaseModel):
    message: Optional[str] = ""
    user:    Optional[str] = ""
    image:   Optional[str] = None

@app.post("/agent")
async def agent_endpoint(payload: Payload):
    print(f"\n📩 [{payload.user}]: {payload.message or '(image)'}")

    thread_id = payload.user or "default"
    config    = {"configurable": {"thread_id": thread_id}}

    try:
        if payload.image:
            human_message = HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{payload.image}"}
                },
                {
                    "type": "text",
                    "text": (f"{payload.message or ''}\n\n"
                             f"Analyse ce recu et extrait: numero camion, tonnage, entreprise. "
                             f"Combine avec les prix fournis et enregistre le voyage.")
                }
            ])
        else:
            human_message = HumanMessage(content=payload.message or "")

        result = graph.invoke(
            {"messages": [SystemMessage(content=SYSTEM_PROMPT), human_message]},
            config=config,
        )

        reply = result["messages"][-1].content
        print(f"📤 BOT: {reply}")

        # ─── DETECTION PDF_READY ──────────────────────────────────────────────
        if "PDF_READY:" in reply:
            # Extraire le premier chemin propre
            pdf_path = reply.split("PDF_READY:")[-1].strip().split("\n")[0].strip()
            return {
                "reply":    "📄 PDF pret, envoi en cours...",
                "pdf_path": pdf_path,
                "user":     payload.user,
            }

        return {"reply": reply}

    except Exception as e:
        print(f"❌ ERREUR: {e}")
        return {"reply": "❌ Erreur interne, reessayez."}

@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("LLM_MODEL", "gpt-4o")}
