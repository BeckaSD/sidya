"""
agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Agent IA + WhatsApp — gestion transport de marchandises.

PIPELINE MODE IMAGE :
  0) Validation prix (stop immédiat si manquant)
  1) ACK WhatsApp  "traitement en cours..."
  2) OCR multithreading → [{id, ocr_text}]
  3) Nettoyage OCR (tagging immatriculation 🔢)
  4) Découpage en lots de 5
  5) Lots → OpenAI en série (prompt + lot JSON)
  6) Fuzzy matching entreprise
  7) save_trip pour chaque reçu valide
  8) Message WhatsApp final récapitulatif

PIPELINE MODE TEXTE :
  1) Chargement mémoire Redis (résumé + 10 derniers msgs)
  2) LangGraph ReAct agent
  3) Sauvegarde échange dans Redis
  4) Réponse
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
from datetime import date
from difflib import get_close_matches, SequenceMatcher
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

# ── Constantes ────────────────────────────────────────────────────────────────

_OPENAI_API_KEY: str   = os.getenv("OPENAI_API_KEY", "")
_LLM_MODEL:      str   = os.getenv("LLM_MODEL", "gpt-4o")

LOT_SIZE:          int   = 5      # reçus par appel OpenAI
OPENAI_TIMEOUT:    float = 45.0   # secondes par lot
FUZZY_THRESHOLD:   float = 0.72   # seuil similarité entreprise [0-1]

# Pattern de détection immatriculation (format maghrébin + européen)
_IMMAT_PATTERN = re.compile(
    r'\b([A-Z]{1,3}[-\s]?\d{3,5}[-\s]?[A-Z]{0,3}'   # EU style
    r'|\d{4,6}[-\s]?[A-Z]{1,3}[-\s]?\d{0,2})\b',      # maghrébin
    re.IGNORECASE,
)

# ── LLM & Agent (branche texte) ───────────────────────────────────────────────

llm = ChatOpenAI(model=_LLM_MODEL, api_key=_OPENAI_API_KEY, temperature=0)

_tools = [
    save_trip, set_price, get_report, get_profit,
    list_entreprises, delete_trip,
    get_report_pdf, get_profit_pdf, get_summary_pdf,
]
_memory = MemorySaver()
graph   = create_react_agent(llm, _tools, checkpointer=_memory)

# ── Prompts ───────────────────────────────────────────────────────────────────

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

_EXTRACT_SYSTEM = """Tu es un extracteur de données de reçus de transport.
Tu reçois une liste JSON de reçus OCR. Pour chaque reçu, extrais les données.

Réponds UNIQUEMENT avec un tableau JSON valide. Format STRICT :
[
  {
    "id": 0,
    "immatriculation": "valeur ou null",
    "tonnage_kg": 12345.6,
    "entreprise": "Nom exact du reçu"
  }
]

Règles :
- Conserver l id de chaque reçu tel quel
- 🔢 dans le texte signale une immatriculation véhicule
- tonnage_kg TOUJOURS en kg (ne pas convertir)
- Si donnée absente ou illisible → null
- Ne jamais inventer de données
- Aucun texte avant ou après le JSON"""


# ── Schéma payload ────────────────────────────────────────────────────────────

class Payload(BaseModel):
    message:     Optional[str]       = ""
    user:        Optional[str]       = ""
    image:       Optional[str]       = None
    images:      Optional[list[str]] = None
    prix_camion: Optional[float]     = None   # OBLIGATOIRE si images
    prix_client: Optional[float]     = None   # OBLIGATOIRE si images


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
    """
    Nettoie le texte OCR :
    - Détecte les immatriculations → préfixe 🔢
    - Supprime les caractères parasites répétés
    """
    if not text:
        return text

    # Tagging immatriculation
    def _tag(m: re.Match) -> str:
        val = m.group(0).upper().replace(" ", "-")
        return f"🔢 {val}"

    text = _IMMAT_PATTERN.sub(_tag, text)

    # Supprime les lignes de moins de 2 caractères (bruit OCR)
    lines = [l for l in text.splitlines() if len(l.strip()) > 2]
    return "\n".join(lines)


def _normalize(name: str) -> str:
    return name.strip().lower()


def _fuzzy_match(raw: str | None, known: list[str]) -> tuple[str | None, float]:
    if not raw or not known:
        return None, 0.0
    norm_map = {_normalize(n): n for n in known}
    matches  = get_close_matches(_normalize(raw), norm_map.keys(), n=1, cutoff=FUZZY_THRESHOLD)
    if matches:
        original = norm_map[matches[0]]
        score    = SequenceMatcher(None, _normalize(raw), matches[0]).ratio()
        return original, score
    return None, 0.0


def _validate_prices(payload: Payload) -> str | None:
    """
    Retourne un message d erreur si les prix sont manquants, None si OK.
    """
    missing: list[str] = []
    if payload.prix_camion is None:
        missing.append("💰 *prix_camion* (prix payé au transporteur par tonne)")
    if payload.prix_client is None:
        missing.append("💵 *prix_client* (prix facturé à l'entreprise par tonne)")
    if missing:
        return (
            "⚠️ *Prix manquants — traitement annulé.*\n\n"
            "Merci de fournir :\n" +
            "\n".join(f"  • {m}" for m in missing) +
            "\n\nRenvoyez vos images avec ces deux prix."
        )
    return None


# ── Appel OpenAI pour un lot ──────────────────────────────────────────────────

def _call_openai_lot(lot: list[dict], lot_num: int) -> list[dict]:
    """
    Envoie un lot de reçus OCR à OpenAI et retourne la liste extraite.
    Appelé en série, un lot après l'autre.
    """
    logger.info("🤖 Lot #%d → OpenAI (%d reçus)", lot_num, len(lot))
    t0 = time.perf_counter()

    user_content = json.dumps(lot, ensure_ascii=False, indent=2)
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
                        {"role": "system", "content": _EXTRACT_SYSTEM},
                        {"role": "user",   "content": user_content},
                    ],
                },
            )
            resp.raise_for_status()
            raw_response = resp.json()["choices"][0]["message"]["content"].strip()

        # Nettoyage éventuel de backticks markdown
        clean = re.sub(r"```(?:json)?|```", "", raw_response).strip()
        data  = json.loads(clean)

        if not isinstance(data, list):
            raise ValueError(f"Réponse OpenAI non-liste : {type(data)}")

        logger.info("✅ Lot #%d traité en %.2fs", lot_num, time.perf_counter() - t0)
        return data

    except json.JSONDecodeError as exc:
        logger.error("❌ Lot #%d JSON invalide : %s | brut=%r", lot_num, exc, raw_response)
        # Retourne des entrées en erreur pour chaque reçu du lot
        return [{"id": r["id"], "erreur": f"JSON invalide : {exc}"} for r in lot]

    except Exception as exc:
        logger.error("❌ Lot #%d erreur OpenAI : %s", lot_num, exc)
        return [{"id": r["id"], "erreur": str(exc)} for r in lot]


# ── Construction récap WhatsApp ───────────────────────────────────────────────

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
            corr   = (
                f"\n   _(corrigé depuis « {r['entreprise_brute']} »)_"
                if r.get("fuzzy_corrige") else ""
            )
            lines.append(f"📋 *Reçu #{num}*")
            lines.append(f"   🚛 Immat      : `{immat}`")
            lines.append(f"   ⚖️  Tonnage    : {tonnes}")
            lines.append(f"   🏢 Entreprise : {ent}{corr}")
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
    logger.info("📩 [%s] images=%d  msg=%r",
                user_id,
                len(payload.images or ([payload.image] if payload.image else [])),
                payload.message)

    try:
        images_bytes = _decode_images(payload)

        # ══════════════════════════════════════════════════════════════════════
        # BRANCHE IMAGE
        # ══════════════════════════════════════════════════════════════════════
        if images_bytes:
            nb = len(images_bytes)

            # ── 0) Validation prix — AVANT TOUT ──────────────────────────────
            price_error = _validate_prices(payload)
            if price_error:
                logger.warning("[%s] Prix manquants — traitement annulé", user_id)
                return {"reply": price_error}

            # ── 1) ACK immédiat WhatsApp ──────────────────────────────────────
            # Note : en production, envoyer ce message via l API WhatsApp
            # avant de continuer le traitement (webhook async).
            ack_message = (
                f"⏳ *Traitement en cours...*\n"
                f"📄 {nb} reçu(s) reçu(s)\n"
                f"💰 Prix camion : {payload.prix_camion} /t\n"
                f"💵 Prix client : {payload.prix_client} /t\n"
                f"Merci de patienter ✨"
            )
            logger.info("📤 ACK → [%s] : %s", user_id, ack_message)

            # ── 2) OCR multithreading (hors LLM) ─────────────────────────────
            logger.info("🖼️  Lancement OCR sur %d image(s)...", nb)
            ocr_results = extract_text_from_images(images_bytes)

            # ── 3) Nettoyage OCR + construction liste JSON ────────────────────
            ocr_list: list[dict] = []
            for idx in sorted(ocr_results):
                entry  = ocr_results[idx]
                if entry["error"]:
                    logger.warning("⚠️  Image #%d OCR échoué : %s", idx + 1, entry["error"])
                    ocr_list.append({
                        "id":       idx,
                        "ocr_text": None,
                        "_ocr_err": entry["error"],
                    })
                else:
                    ocr_list.append({
                        "id":       idx,
                        "ocr_text": _clean_ocr(entry["text"] or ""),
                    })

            # ── 4) Entreprises connues pour fuzzy ─────────────────────────────
            try:
                known_raw   = list_entreprises.invoke({})
                known_names: list[str] = (
                    known_raw if isinstance(known_raw, list)
                    else [l.strip() for l in str(known_raw).splitlines() if l.strip()]
                )
            except Exception as exc:
                logger.warning("list_entreprises indisponible : %s", exc)
                known_names = []

            # ── 5) Découpage en lots de LOT_SIZE ─────────────────────────────
            # On exclut les reçus en erreur OCR du traitement OpenAI
            valid_ocr  = [r for r in ocr_list if r.get("ocr_text")]
            failed_ocr = {r["id"]: r["_ocr_err"] for r in ocr_list if not r.get("ocr_text")}

            lots = [
                valid_ocr[i: i + LOT_SIZE]
                for i in range(0, len(valid_ocr), LOT_SIZE)
            ]

            logger.info(
                "📦 %d reçu(s) valides → %d lot(s) de max %d",
                len(valid_ocr), len(lots), LOT_SIZE,
            )

            # ── 6) Appels OpenAI en SÉRIE (lot par lot) ───────────────────────
            all_extracted: list[dict] = []
            for lot_num, lot in enumerate(lots, start=1):
                # Envoi uniquement des champs utiles à OpenAI
                lot_payload = [{"id": r["id"], "ocr_text": r["ocr_text"]} for r in lot]
                extracted   = _call_openai_lot(lot_payload, lot_num)
                all_extracted.extend(extracted)

            # ── 7) Fuzzy matching + save_trip ─────────────────────────────────
            results_map: dict[int, dict] = {}

            # Initialiser avec les erreurs OCR
            for idx, err in failed_ocr.items():
                results_map[idx] = {
                    "id":     idx,
                    "saved":  False,
                    "erreur": f"OCR échoué : {err}",
                }

            for item in all_extracted:
                idx = item.get("id", -1)
                entry: dict[str, Any] = {
                    "id":              idx,
                    "saved":           False,
                    "fuzzy_corrige":   False,
                    "immatriculation": None,
                    "tonnage_kg":      None,
                    "entreprise":      None,
                    "entreprise_brute": None,
                    "erreur":          item.get("erreur"),
                }

                if item.get("erreur"):
                    results_map[idx] = entry
                    continue

                immat      = item.get("immatriculation")
                tonnage_kg = item.get("tonnage_kg")
                ent_brute  = item.get("entreprise")

                entry["immatriculation"]   = immat
                entry["tonnage_kg"]        = tonnage_kg
                entry["entreprise_brute"]  = ent_brute

                # Fuzzy matching
                ent_corrige, score = _fuzzy_match(ent_brute, known_names)
                if ent_corrige and _normalize(ent_corrige) != _normalize(ent_brute or ""):
                    entry["entreprise"]    = ent_corrige
                    entry["fuzzy_corrige"] = True
                    logger.info("🔧 Reçu #%d  '%s' → '%s' (%.0f%%)",
                                idx + 1, ent_brute, ent_corrige, score * 100)
                else:
                    entry["entreprise"] = ent_corrige or ent_brute
                    if not ent_corrige and ent_brute and known_names:
                        logger.warning("⚠️  Reçu #%d '%s' : aucun match fuzzy", idx + 1, ent_brute)

                # Validation
                if not tonnage_kg:
                    entry["erreur"] = "Tonnage introuvable"
                    results_map[idx] = entry
                    continue
                if not entry["entreprise"]:
                    entry["erreur"] = "Entreprise introuvable"
                    results_map[idx] = entry
                    continue

                # Enregistrement
                try:
                    save_kwargs: dict[str, Any] = {
                        "entreprise":  entry["entreprise"],
                        "tonnage":     tonnage_kg,
                        "unite":       "kg",
                        "prix_camion": payload.prix_camion,
                        "prix_client": payload.prix_client,
                    }
                    if immat:
                        save_kwargs["camion"] = immat

                    save_trip.invoke(save_kwargs)
                    entry["saved"] = True
                    logger.info(
                        "💾 Reçu #%d enregistré — %s / %s / %.0f kg",
                        idx + 1, immat, entry["entreprise"], tonnage_kg,
                    )
                except Exception as exc:
                    entry["erreur"] = f"Erreur save_trip : {exc}"
                    logger.error("❌ Reçu #%d save_trip : %s", idx + 1, exc)

                results_map[idx] = entry

            # ── 8) Message final récapitulatif ────────────────────────────────
            all_results = list(results_map.values())
            recap = _build_recap(
                all_results,
                nb_total    = nb,
                prix_camion = payload.prix_camion,
                prix_client = payload.prix_client,
            )
            logger.info("📤 Récap final :\n%s", recap)

            return {
                "ack":   ack_message,   # à envoyer immédiatement via webhook
                "reply": recap,         # réponse finale
            }

        # ══════════════════════════════════════════════════════════════════════
        # BRANCHE TEXTE — LangGraph + mémoire Redis
        # ══════════════════════════════════════════════════════════════════════
        else:
            user_msg = payload.message or ""

            # Chargement mémoire Redis
            summary, history = load_context(user_id)
            memory_block     = build_memory_prompt(summary, history)

            # Prompt système enrichi avec la mémoire
            system_content = _BASE_SYSTEM
            if memory_block:
                system_content += f"\n\n{memory_block}"

            config = {"configurable": {"thread_id": user_id}}
            result = graph.invoke(
                {
                    "messages": [
                        SystemMessage(content=system_content),
                        HumanMessage(content=user_msg),
                    ]
                },
                config=config,
            )

            reply = result["messages"][-1].content
            logger.info("📤 BOT [%s]: %s", user_id, reply)

            # Sauvegarde échange dans Redis
            save_exchange(user_id, user_msg, reply)

            # Détection PDF_READY
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
        "status":          "ok",
        "model":           _LLM_MODEL,
        "lot_size":        LOT_SIZE,
        "openai_timeout":  OPENAI_TIMEOUT,
        "fuzzy_threshold": FUZZY_THRESHOLD,
    }


@app.delete("/memory/{user_id}")
async def reset_memory(user_id: str):
    """Efface la mémoire conversationnelle d'un utilisateur."""
    from memory_service import clear_memory
    clear_memory(user_id)
    return {"status": "ok", "message": f"Mémoire effacée pour {user_id}"}