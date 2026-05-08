"""
agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Agent IA + WhatsApp — gestion transport de marchandises.

PIPELINE MODE IMAGE :
  0) LLM extrait prix_camion + prix_client depuis le message texte
     → flexible : "5000 7000", "سعر الشاحن 5000 سعر الشركة 7000",
                  "camion=5000 client=7000", deux chiffres seuls, etc.
     → si manquant → demande UNE SEULE FOIS (pas une fois par image)
  1) ACK WhatsApp immédiat
  2) OCR multithreading → [{id, ocr_text}]
  3) Nettoyage OCR (tagging immatriculation 🔢)
  4) Découpage en lots de LOT_SIZE
  5) Lots → OpenAI en série
     → le LLM gère aussi le fuzzy matching entreprise dans le prompt
  6) save_trip pour chaque reçu valide
  7) Message WhatsApp final récapitulatif

PIPELINE MODE TEXTE :
  1) Chargement mémoire Redis (résumé + 10 derniers msgs)
  2) LangGraph ReAct agent
  3) Sauvegarde échange dans Redis
  4) Réponse
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from datetime import date
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from memory_service import build_memory_prompt, load_context, save_exchange
from ocr_service import extract_text_from_images
from tools import (
    delete_trip,
    get_profit,
    get_profit_pdf,
    get_report,
    get_report_pdf,
    get_summary_pdf,
    list_entreprises,
    save_trip,
    set_price,
)

# ── Init ──────────────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("agent")

app = FastAPI()

# ── Diagnostic au démarrage : affiche le schéma réel de save_trip ─────────────

@app.on_event("startup")
async def _startup_diagnostics():
    try:
        schema = save_trip.args_schema.schema()
        import json as _json
        print("\n" + "═"*60)
        print("🔍 DIAGNOSTIC save_trip — paramètres attendus :")
        print(_json.dumps(schema, indent=2, ensure_ascii=False))
        print("═"*60 + "\n")
    except Exception as e:
        print(f"⚠️  Impossible de lire le schéma save_trip : {e}")

# ── Constantes ────────────────────────────────────────────────────────────────

_OPENAI_API_KEY: str   = os.getenv("OPENAI_API_KEY", "")
_LLM_MODEL:      str   = os.getenv("LLM_MODEL", "gpt-4o")

LOT_SIZE:        int   = 5     # reçus par appel OpenAI
OPENAI_TIMEOUT:  float = 45.0  # secondes par lot

# ── LLM & Agent (branche texte) ───────────────────────────────────────────────

llm = ChatOpenAI(model=_LLM_MODEL, api_key=_OPENAI_API_KEY, temperature=0)

_tools = [
    save_trip, set_price, get_report, get_profit,
    list_entreprises, delete_trip,
    get_report_pdf, get_profit_pdf, get_summary_pdf,
]
_memory = MemorySaver()
graph   = create_react_agent(llm, _tools, checkpointer=_memory)

# ── Prompt système agent texte ────────────────────────────────────────────────

_BASE_SYSTEM = f"""Tu es un assistant de gestion de transport de marchandises.
La date du jour est : {date.today().strftime('%d/%m/%Y')}.
Le mois actuel : {date.today().month} — Année : {date.today().year}.

Tu aides a:
- Identifier dans les reçus OCR : immatriculation (🔢), tonnage, entreprise
- Le tonnage est TOUJOURS en KG — passer unite="kg" a save_trip
- Enregistrer, modifier, supprimer des voyages
- Générer rapports et bénéfices par entreprise (texte ou PDF)
- Générer un PDF global avec get_summary_pdf

Règles :
- Répondre dans la langue de l utilisateur
- Mois/année actuels si non précisés
- Nom entreprise inexact → choisir le plus proche automatiquement
- Prix camion = payé au transporteur/tonne | Prix client = facturé à l entreprise/tonne

PDF :
- Une entreprise rapport  → get_report_pdf
- Une entreprise bénéfice → get_profit_pdf
- Global/synthèse         → get_summary_pdf (UN SEUL appel)
- Réponse si PDF : UNIQUEMENT PDF_READY:<chemin>
- Sois concis, utilise les emojis WhatsApp
"""

# ── Prompt extraction prix (appelé UNE seule fois avant le pipeline) ──────────

_PRICE_EXTRACT_SYSTEM = """Tu es un extracteur de prix depuis un message WhatsApp.
L utilisateur envoie un message qui contient deux prix :
- prix_camion : ce que l on paie au transporteur par tonne
- prix_client : ce que l entreprise paie par tonne

Le message peut être dans n importe quelle langue et n importe quel format :
exemples : "5000 7000", "camion 5000 client 7000", "سعر الشاحن 5000 سعر الشركة 7000",
           "transport=5000 entreprise=7000", "le petit est camion le grand est client"

Règle : le prix_camion est TOUJOURS inférieur au prix_client.
Si un seul chiffre est fourni ou les deux sont identiques, retourner null pour les deux.

Réponds UNIQUEMENT avec un objet JSON valide, sans markdown :
{"prix_camion": 5000.0, "prix_client": 7000.0}
ou si impossible :
{"prix_camion": null, "prix_client": null}"""

# ── Prompt extraction reçus OCR (avec fuzzy entreprise intégré) ───────────────

def _build_extract_system(known_names: list[str]) -> str:
    """
    Prompt d extraction des reçus OCR.
    Intègre la liste des entreprises connues pour que le LLM fasse
    lui-même le matching — pas besoin de fuzzy côté code.
    """
    entreprises_block = ""
    if known_names:
        liste = "\n".join(f"  - {n}" for n in known_names)
        entreprises_block = f"""
Entreprises connues dans la base de données :
{liste}

Règle entreprise : si le nom lu sur le reçu ressemble à l une des entreprises
connues (faute d orthographe, abréviation, etc.), utiliser le nom exact de la base.
Exemples : "achemin" → "Achemine", "somine" → "Somine SA", etc.
Si aucun match raisonnable → retourner le nom tel quel du reçu.
"""

    return f"""Tu es un extracteur de données de reçus de transport.
Tu reçois une liste JSON de reçus OCR. Pour chaque reçu, extrais les données.
{entreprises_block}
Réponds UNIQUEMENT avec un tableau JSON valide. Format STRICT :
[
  {{
    "id": 0,
    "immatriculation": "valeur ou null",
    "tonnage_kg": 12345.6,
    "entreprise": "Nom exact de la base ou du reçu"
  }}
]

Règles :
- Conserver l id de chaque reçu tel quel
- 🔢 dans le texte signale une immatriculation véhicule
- tonnage_kg TOUJOURS en kg, ne pas convertir
- Si donnée absente ou illisible → null
- Ne jamais inventer de données
- Aucun texte avant ou après le JSON"""


# ── Schéma payload ────────────────────────────────────────────────────────────

class Payload(BaseModel):
    message: Optional[str]       = ""
    user:    Optional[str]       = ""
    image:   Optional[str]       = None
    images:  Optional[list[str]] = None
    # Les prix peuvent aussi arriver directs depuis un client API avancé
    # mais le LLM les extrait aussi depuis message — les deux sont supportés
    prix_camion: Optional[float] = None
    prix_client: Optional[float] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_images(payload: Payload) -> list[bytes]:
    raw = payload.images or ([payload.image] if payload.image else [])
    decoded: list[bytes] = []
    for b64 in raw:
        try:
            decoded.append(base64.b64decode(b64))
        except Exception as exc:
            logger.warning("Décodage base64 échoué : %s", exc)
    return decoded


def _clean_ocr(text: str) -> str:
    """Nettoie le texte OCR et tague les immatriculations avec 🔢."""
    if not text:
        return ""
    _IMMAT = re.compile(
        r'\b([A-Z]{1,3}[-\s]?\d{3,5}[-\s]?[A-Z]{0,3}|\d{4,6}[-\s]?[A-Z]{1,3}[-\s]?\d{0,2})\b',
        re.IGNORECASE,
    )
    def _tag(m: re.Match) -> str:
        return f"🔢 {m.group(0).upper().replace(' ', '-')}"

    text  = _IMMAT.sub(_tag, text)
    lines = [l for l in text.splitlines() if len(l.strip()) > 2]
    return "\n".join(lines)


def _llm_extract_prices(message: str) -> dict:
    """
    Appelle OpenAI pour extraire prix_camion et prix_client
    depuis le message texte de l utilisateur.
    Retourne {"prix_camion": float|None, "prix_client": float|None}.
    """
    if not message or not message.strip():
        return {"prix_camion": None, "prix_client": None}

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {_OPENAI_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       _LLM_MODEL,
                    "temperature": 0,
                    "max_tokens":  60,
                    "messages": [
                        {"role": "system", "content": _PRICE_EXTRACT_SYSTEM},
                        {"role": "user",   "content": message},
                    ],
                },
            )
            resp.raise_for_status()
            raw  = resp.json()["choices"][0]["message"]["content"].strip()
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            data  = json.loads(clean)
            logger.info("💲 Prix extraits par LLM : %s", data)
            return data
    except Exception as exc:
        logger.error("Erreur extraction prix LLM : %s", exc)
        return {"prix_camion": None, "prix_client": None}


def _call_openai_lot(lot: list[dict], lot_num: int, extract_system: str) -> list[dict]:
    """
    Envoie un lot de reçus OCR à OpenAI et retourne la liste extraite.
    Appelé en série, un lot après l autre.
    """
    logger.info("🤖 Lot #%d → OpenAI (%d reçus)", lot_num, len(lot))
    t0           = time.perf_counter()
    raw_response = ""

    try:
        with httpx.Client(timeout=OPENAI_TIMEOUT) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {_OPENAI_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       _LLM_MODEL,
                    "temperature": 0,
                    "max_tokens":  600,
                    "messages": [
                        {"role": "system", "content": extract_system},
                        {"role": "user",   "content": json.dumps(lot, ensure_ascii=False, indent=2)},
                    ],
                },
            )
            resp.raise_for_status()
            raw_response = resp.json()["choices"][0]["message"]["content"].strip()

        clean = re.sub(r"```(?:json)?|```", "", raw_response).strip()
        data  = json.loads(clean)

        if not isinstance(data, list):
            raise ValueError(f"Réponse non-liste : {type(data)}")

        logger.info("✅ Lot #%d — %.2fs", lot_num, time.perf_counter() - t0)
        return data

    except json.JSONDecodeError as exc:
        logger.error("❌ Lot #%d JSON invalide : %s | brut=%r", lot_num, exc, raw_response)
        return [{"id": r["id"], "erreur": f"JSON invalide : {exc}"} for r in lot]
    except Exception as exc:
        logger.error("❌ Lot #%d erreur : %s", lot_num, exc)
        return [{"id": r["id"], "erreur": str(exc)} for r in lot]


def _build_recap(
    results:     list[dict],
    nb_total:    int,
    prix_camion: float,
    prix_client: float,
) -> str:
    nb_ok  = sum(1 for r in results if r.get("saved"))
    nb_err = nb_total - nb_ok

    lines = [
        f"✅ *Traitement terminé — {nb_ok}/{nb_total} reçu(s) enregistré(s)*",
        f"💰 Prix camion : {prix_camion} /t  |  💵 Prix client : {prix_client} /t",
        "",
    ]

    for r in sorted(results, key=lambda x: x["id"]):
        num = r["id"] + 1
        if r.get("saved"):
            immat  = r.get("immatriculation") or "—"
            kg     = r.get("tonnage_kg")
            tonnes = f"{kg / 1000:.3f} t  ({kg:,.0f} kg)" if kg else "—"
            ent    = r.get("entreprise") or "—"
            lines.append(f"📋 *Reçu #{num}*")
            lines.append(f"   🚛 Immat      : `{immat}`")
            lines.append(f"   ⚖️  Tonnage    : {tonnes}")
            lines.append(f"   🏢 Entreprise : {ent}")
        else:
            err = r.get("erreur") or "Données manquantes"
            lines.append(f"❌ *Reçu #{num}* — non enregistré")
            lines.append(f"   ↳ {err}")
        lines.append("")

    if nb_err:
        lines.append(f"⚠️ {nb_err} reçu(s) ignoré(s) — vérifiez la qualité des images.")

    return "\n".join(lines).strip()


# ── Endpoint principal ────────────────────────────────────────────────────────

@app.post("/agent")
async def agent_endpoint(payload: Payload):
    user_id = payload.user or "default"
    nb_imgs = len(payload.images or ([payload.image] if payload.image else []))
    logger.info("📩 [%s] images=%d  msg=%r", user_id, nb_imgs, payload.message)

    try:
        images_bytes = _decode_images(payload)

        # ══════════════════════════════════════════════════════════════════════
        # BRANCHE IMAGE
        # ══════════════════════════════════════════════════════════════════════
        if images_bytes:
            nb = len(images_bytes)

            # ── 0) Extraction prix par LLM — UNE seule fois ───────────────────
            # Priorité : valeurs JSON explicites > extraction depuis message
            prix_camion = payload.prix_camion
            prix_client = payload.prix_client

            if prix_camion is None or prix_client is None:
                extracted_prices = _llm_extract_prices(payload.message or "")
                prix_camion = prix_camion or extracted_prices.get("prix_camion")
                prix_client = prix_client or extracted_prices.get("prix_client")

            # Si encore manquant → demander UNE SEULE FOIS (indépendant du nb d images)
            missing = []
            if prix_camion is None:
                missing.append("💰 *prix camion* (payé au transporteur par tonne)")
            if prix_client is None:
                missing.append("💵 *prix client* (facturé à l'entreprise par tonne)")

            if missing:
                logger.warning("[%s] Prix manquants après extraction LLM", user_id)
                return {
                    "reply": (
                        "⚠️ *Prix manquants — merci de préciser :*\n\n" +
                        "\n".join(f"  • {m}" for m in missing) +
                        "\n\nExemple : _5000 7000_ ou _camion 5000 client 7000_\n"
                        "Renvoyez vos images avec ces prix."
                    )
                }

            logger.info("💲 [%s] prix_camion=%.0f  prix_client=%.0f", user_id, prix_camion, prix_client)

            # ── 1) ACK immédiat ───────────────────────────────────────────────
            ack_message = (
                f"⏳ *Traitement en cours...*\n"
                f"📄 {nb} reçu(s) reçu(s)\n"
                f"💰 Prix camion : {prix_camion} /t\n"
                f"💵 Prix client : {prix_client} /t\n"
                f"Merci de patienter ✨"
            )
            logger.info("📤 ACK → [%s]", user_id)

            # ── 2) OCR multithreading ─────────────────────────────────────────
            logger.info("🖼️  OCR sur %d image(s)...", nb)
            ocr_results = extract_text_from_images(images_bytes)

            # ── 3) Nettoyage OCR ──────────────────────────────────────────────
            ocr_list:   list[dict] = []
            failed_ocr: dict[int, str] = {}

            for idx in sorted(ocr_results):
                entry = ocr_results[idx]
                if entry["error"]:
                    logger.warning("⚠️  Image #%d OCR échoué : %s", idx + 1, entry["error"])
                    failed_ocr[idx] = entry["error"]
                else:
                    ocr_list.append({
                        "id":       idx,
                        "ocr_text": _clean_ocr(entry["text"] or ""),
                    })

            # ── 4) Entreprises connues → données au LLM pour fuzzy ────────────
            try:
                known_raw   = list_entreprises.invoke({})
                known_names: list[str] = (
                    known_raw if isinstance(known_raw, list)
                    else [l.strip() for l in str(known_raw).splitlines() if l.strip()]
                )
            except Exception as exc:
                logger.warning("list_entreprises indisponible : %s", exc)
                known_names = []

            # Prompt extraction avec liste entreprises intégrée
            extract_system = _build_extract_system(known_names)

            # ── 5) Lots de LOT_SIZE → OpenAI en série ─────────────────────────
            lots = [ocr_list[i: i + LOT_SIZE] for i in range(0, len(ocr_list), LOT_SIZE)]
            logger.info("📦 %d reçu(s) → %d lot(s)", len(ocr_list), len(lots))

            all_extracted: list[dict] = []
            for lot_num, lot in enumerate(lots, start=1):
                extracted = _call_openai_lot(lot, lot_num, extract_system)
                all_extracted.extend(extracted)

            # ── 6) save_trip pour chaque reçu valide ──────────────────────────
            results_map: dict[int, dict] = {}

            # Erreurs OCR
            for idx, err in failed_ocr.items():
                results_map[idx] = {"id": idx, "saved": False, "erreur": f"OCR échoué : {err}"}

            for item in all_extracted:
                idx   = item.get("id", -1)
                entry: dict[str, Any] = {
                    "id":              idx,
                    "saved":           False,
                    "immatriculation": None,
                    "tonnage_kg":      None,
                    "entreprise":      None,
                    "erreur":          item.get("erreur"),
                }

                if item.get("erreur"):
                    results_map[idx] = entry
                    continue

                immat      = item.get("immatriculation")
                tonnage_kg = item.get("tonnage_kg")
                entreprise = item.get("entreprise")

                entry["immatriculation"] = immat
                entry["tonnage_kg"]      = tonnage_kg
                entry["entreprise"]      = entreprise

                if not tonnage_kg:
                    entry["erreur"] = "Tonnage introuvable"
                    results_map[idx] = entry
                    continue
                if not entreprise:
                    entry["erreur"] = "Entreprise introuvable"
                    results_map[idx] = entry
                    continue

                try:
                    # ── Préparation save_kwargs ───────────────────────────────
                    # DIAGNOSTIC : on inspecte le schéma réel au 1er appel
                    # pour trouver le bon nom du champ immatriculation
                    try:
                        _schema     = save_trip.args_schema.schema()
                        _props      = _schema.get("properties", {})
                        _req        = _schema.get("required", [])
                        # Cherche le champ qui ressemble à "camion" / "num_camion" / "immat"
                        _immat_key  = next(
                            (k for k in _props
                             if any(x in k.lower() for x in ["camion", "immat", "num", "vehic", "truck"])),
                            None,
                        )
                        print(f"\n🔍 save_trip properties : {list(_props.keys())}")
                        print(f"🔍 save_trip required   : {_req}")
                        print(f"🔍 champ immat détecté  : {_immat_key}\n")
                    except Exception as _e:
                        print(f"⚠️  Diagnostic schéma échoué : {_e}")
                        _immat_key = "num_camion"   # fallback conservateur

                    save_kwargs: dict[str, Any] = {
                        "entreprise":  entreprise,
                        "tonnage":     tonnage_kg,
                        "unite":       "kg",
                        "prix_camion": prix_camion,
                        "prix_client": prix_client,
                    }
                    # Ajout immatriculation avec le bon nom de champ
                    if immat and _immat_key:
                        save_kwargs[_immat_key] = immat

                    print(f"📤 save_trip invoke #{idx+1} → {save_kwargs}")
                    save_trip.invoke(save_kwargs)
                    entry["saved"] = True
                    logger.info("💾 #%d — %s / %s / %.0f kg", idx + 1, immat, entreprise, tonnage_kg)
                    print(f"✅ save_trip #{idx+1} OK")

                except Exception as exc:
                    entry["erreur"] = f"Erreur save_trip : {exc}"
                    logger.error("❌ Reçu #%d save_trip : %s", idx + 1, exc)
                    print(f"❌ save_trip #{idx+1} ERREUR : {exc}")

                results_map[idx] = entry

            # ── 7) Récap final ─────────────────────────────────────────────────
            all_results = list(results_map.values())
            recap = _build_recap(all_results, nb_total=nb,
                                 prix_camion=prix_camion, prix_client=prix_client)
            logger.info("📤 Récap :\n%s", recap)

            return {"ack": ack_message, "reply": recap}

        # ══════════════════════════════════════════════════════════════════════
        # BRANCHE TEXTE — LangGraph + mémoire Redis
        # ══════════════════════════════════════════════════════════════════════
        else:
            user_msg = payload.message or ""

            summary, history = load_context(user_id)
            memory_block     = build_memory_prompt(summary, history)

            system_content = _BASE_SYSTEM
            if memory_block:
                system_content += f"\n\n{memory_block}"

            config = {"configurable": {"thread_id": user_id}}
            result = graph.invoke(
                {"messages": [
                    SystemMessage(content=system_content),
                    HumanMessage(content=user_msg),
                ]},
                config=config,
            )

            reply = result["messages"][-1].content
            logger.info("📤 BOT [%s]: %s", user_id, reply)

            save_exchange(user_id, user_msg, reply)

            if "PDF_READY:" in reply:
                pdf_path = reply.split("PDF_READY:")[-1].strip().split("\n")[0].strip()
                return {
                    "reply":    "📄 PDF prêt, envoi en cours...",
                    "pdf_path": pdf_path,
                    "user":     user_id,
                }

            return {"reply": reply}

    except Exception as exc:
        logger.exception("❌ ERREUR agent [%s]: %s", user_id, exc)
        return {"reply": "❌ Erreur interne, réessayez."}


# ── Endpoints utilitaires ─────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":         "ok",
        "model":          _LLM_MODEL,
        "lot_size":       LOT_SIZE,
        "openai_timeout": OPENAI_TIMEOUT,
    }


@app.delete("/memory/{user_id}")
async def reset_memory(user_id: str):
    from memory_service import clear_memory
    clear_memory(user_id)
    return {"status": "ok", "message": f"Mémoire effacée pour {user_id}"}