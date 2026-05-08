"""
agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Agent IA + WhatsApp — gestion transport de marchandises.

PIPELINE MODE IMAGE :
  0)  Extraction prix depuis message (LLM) + cache Redis 5min
      → si manquant : demande UNE SEULE FOIS

  1)  Groupement WhatsApp (Redis) :
      Chaque webhook ajoute son image dans un buffer Redis.
      Le 1er webhook démarre un timer de GROUP_WAIT_SEC secondes.
      Quand le timer expire → traitement de TOUTES les images groupées.
      → UNE SEULE réponse finale, peu importe le nombre d images.

  2)  ACK immédiat au 1er webhook seulement

  3)  OCR multithreading (RapidOCR, 4 pipelines cascade)
      → images avec immat trouvée : OK
      → images SANS immat : marquées pour fallback OpenAI vision

  4)  Fallback OpenAI vision (GPT-4o) pour les images difficiles
      → envoi de l image base64 directement à OpenAI
      → extraction directe : immat + tonnage + entreprise

  5)  Lots de LOT_SIZE → OpenAI texte en série (images OCR valides)

  6)  save_trip pour chaque reçu valide

  7)  UNE SEULE réponse WhatsApp récapitulative

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
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from memory_service import build_memory_prompt, load_context, save_exchange, get_redis
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

LOT_SIZE:        int   = 5     # reçus OCR par appel OpenAI texte
OPENAI_TIMEOUT:  float = 60.0  # secondes par lot

# Groupement WhatsApp : délai d attente avant de traiter le groupe
# WhatsApp envoie les images d un même envoi en ~1-2 secondes
GROUP_WAIT_SEC:  int   = 8

# TTL cache prix Redis (secondes)
PRIX_TTL_SEC:    int   = 300   # 5 minutes

# Pattern immatriculation mauritanienne : 4 chiffres + 2 lettres + 2 chiffres
_IMMAT_RE = re.compile(r'\b(\d{4}[A-Z]{2}\d{2})\b', re.IGNORECASE)

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
- Identifier dans les reçus : immatriculation, tonnage, entreprise
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

_PRICE_EXTRACT_SYSTEM = """Tu es un extracteur de prix depuis un message WhatsApp.
L utilisateur envoie un message qui contient deux prix :
- prix_camion : ce que l on paie au transporteur par tonne
- prix_client : ce que l entreprise paie par tonne

Formats possibles (toute langue) :
"5000 7000", "camion 5000 client 7000", "سعر الشاحن 5000 سعر الشركة 7000",
"transport=5000 entreprise=7000", deux chiffres seuls dans n importe quel ordre.

Règle : prix_camion < prix_client toujours.
Si un seul chiffre ou les deux sont identiques → null pour les deux.

Réponds UNIQUEMENT JSON sans markdown :
{"prix_camion": 5000.0, "prix_client": 7000.0}
ou {"prix_camion": null, "prix_client": null}"""


_VISION_EXTRACT_SYSTEM = """Tu es un extracteur de données de reçu de transport.
Tu reçois une image de reçu. Extrais exactement ces données.

Réponds UNIQUEMENT avec un objet JSON valide, sans markdown :
{
  "immatriculation": "XXXXXX ou null",
  "tonnage_kg": 12345.6,
  "entreprise": "Nom exact de l entreprise"
}

Règles :
- immatriculation : format mauritanien 4 chiffres + 2 lettres + 2 chiffres (ex: 3169AA11)
- tonnage_kg : TOUJOURS en kg, ne pas convertir
- entreprise : nom exact tel qu il apparaît sur le reçu
- Si donnée absente → null
- Ne jamais inventer"""


def _build_extract_system(known_names: list[str]) -> str:
    """Prompt extraction OCR avec liste entreprises pour fuzzy LLM."""
    block = ""
    if known_names:
        liste = "\n".join(f"  - {n}" for n in known_names)
        block = f"""
Entreprises connues dans la base :
{liste}

Règle : si le nom du reçu ressemble à une entreprise connue (faute, abréviation),
utiliser le nom exact de la base. Sinon retourner le nom du reçu tel quel.
"""
    return f"""Tu es un extracteur de données de reçus de transport.
Tu reçois une liste JSON de reçus OCR. Pour chaque reçu, extrais les données.
{block}
Réponds UNIQUEMENT avec un tableau JSON valide. Format STRICT :
[
  {{
    "id": 0,
    "immatriculation": "valeur ou null",
    "tonnage_kg": 12345.6,
    "entreprise": "Nom exact"
  }}
]

Règles :
- Conserver l id tel quel
- 🔢 signale une immatriculation véhicule
- tonnage_kg TOUJOURS en kg
- Donnée absente/illisible → null
- Aucun texte avant ou après le JSON"""


# ── Schéma payload ────────────────────────────────────────────────────────────

class Payload(BaseModel):
    message:     Optional[str]       = ""
    user:        Optional[str]       = ""
    image:       Optional[str]       = None   # 1 image base64
    images:      Optional[list[str]] = None   # N images base64
    prix_camion: Optional[float]     = None
    prix_client: Optional[float]     = None


# ── Helpers base ──────────────────────────────────────────────────────────────

def _decode_images(payload: Payload) -> list[bytes]:
    raw = payload.images or ([payload.image] if payload.image else [])
    out: list[bytes] = []
    for b64 in raw:
        try:
            out.append(base64.b64decode(b64))
        except Exception as exc:
            logger.warning("Décodage base64 échoué : %s", exc)
    return out


def _has_immat(text: str) -> bool:
    """Vérifie si une immatriculation mauritanienne est présente dans le texte OCR."""
    return bool(_IMMAT_RE.search(text)) if text else False


def _clean_ocr(text: str) -> str:
    """Tague les immatriculations avec 🔢 et filtre les lignes trop courtes."""
    if not text:
        return ""
    def _tag(m: re.Match) -> str:
        return f"🔢 {m.group(0).upper()}"
    text  = _IMMAT_RE.sub(_tag, text)
    lines = [l for l in text.splitlines() if len(l.strip()) > 2]
    return "\n".join(lines)


def _openai_headers() -> dict:
    return {
        "Authorization": f"Bearer {_OPENAI_API_KEY}",
        "Content-Type":  "application/json",
    }


# ── Extraction prix ───────────────────────────────────────────────────────────

def _llm_extract_prices(message: str) -> dict:
    if not message or not message.strip():
        return {"prix_camion": None, "prix_client": None}
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=_openai_headers(),
                json={
                    "model": _LLM_MODEL, "temperature": 0, "max_tokens": 60,
                    "messages": [
                        {"role": "system", "content": _PRICE_EXTRACT_SYSTEM},
                        {"role": "user",   "content": message},
                    ],
                },
            )
            resp.raise_for_status()
            raw   = resp.json()["choices"][0]["message"]["content"].strip()
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            data  = json.loads(clean)
            logger.info("💲 Prix extraits : %s", data)
            return data
    except Exception as exc:
        logger.error("Erreur extraction prix : %s", exc)
        return {"prix_camion": None, "prix_client": None}


# ── Groupement WhatsApp (Redis) ───────────────────────────────────────────────
#
# Clés Redis par user_id :
#   group:{user_id}:images   → liste de base64 (LPUSH)
#   group:{user_id}:prix     → JSON {prix_camion, prix_client}
#   group:{user_id}:lock     → verrou de traitement (SETNX)
#   group:{user_id}:timer    → marqueur "timer démarré" (SETNX)
# ─────────────────────────────────────────────────────────────────────────────

def _redis_group_add(user_id: str, b64_images: list[str], prix: dict | None, message: str) -> bool:
    """
    Ajoute les images au buffer groupe Redis.
    Retourne True si c est le PREMIER webhook du groupe (doit démarrer le timer).
    """
    r = get_redis()
    pipe = r.pipeline()

    key_imgs  = f"group:{user_id}:images"
    key_prix  = f"group:{user_id}:prix"
    key_timer = f"group:{user_id}:timer"
    key_msg   = f"group:{user_id}:message"

    # Ajout des images (chacune comme élément séparé)
    for b64 in b64_images:
        pipe.rpush(key_imgs, b64)

    # TTL sécurité : auto-expiration si le timer ne se déclenche jamais
    pipe.expire(key_imgs, GROUP_WAIT_SEC + 30)

    # Sauvegarde du message texte (pour extraction prix)
    if message:
        pipe.set(key_msg, message, ex=GROUP_WAIT_SEC + 30)

    # Sauvegarde des prix si présents
    if prix and prix.get("prix_camion") and prix.get("prix_client"):
        pipe.set(key_prix, json.dumps(prix), ex=PRIX_TTL_SEC)

    pipe.execute()

    # SETNX sur le timer : retourne True seulement pour le 1er webhook
    is_first = r.setnx(key_timer, "1")
    if is_first:
        r.expire(key_timer, GROUP_WAIT_SEC + 30)

    nb = r.llen(key_imgs)
    logger.info("📥 [%s] +%d image(s) → buffer=%d | premier=%s",
                user_id, len(b64_images), nb, is_first)
    return bool(is_first)


def _redis_group_collect(user_id: str) -> tuple[list[bytes], dict | None]:
    """
    Récupère et vide atomiquement le buffer groupe.
    Retourne (images_bytes, prix_dict).
    """
    r = get_redis()
    key_imgs  = f"group:{user_id}:images"
    key_prix  = f"group:{user_id}:prix"
    key_lock  = f"group:{user_id}:lock"
    key_timer = f"group:{user_id}:timer"
    key_msg   = f"group:{user_id}:message"

    # Verrou : s assure qu un seul coroutine traite ce groupe
    acquired = r.setnx(key_lock, "1")
    if not acquired:
        logger.info("🔒 [%s] Groupe déjà en cours de traitement", user_id)
        return [], None
    r.expire(key_lock, 120)

    # Récupération atomique
    pipe = r.pipeline()
    pipe.lrange(key_imgs, 0, -1)
    pipe.get(key_prix)
    pipe.get(key_msg)
    pipe.delete(key_imgs, key_prix, key_timer, key_lock, key_msg)
    results = pipe.execute()

    raw_b64s   = results[0] or []
    prix_json  = results[1]
    msg_cached = results[2]

    # Décodage images
    images_bytes: list[bytes] = []
    for b64 in raw_b64s:
        try:
            images_bytes.append(base64.b64decode(b64))
        except Exception:
            pass

    # Prix : depuis Redis ou extraction LLM depuis message en cache
    prix: dict | None = None
    if prix_json:
        try:
            prix = json.loads(prix_json)
        except Exception:
            pass

    if not prix and msg_cached:
        extracted = _llm_extract_prices(msg_cached)
        if extracted.get("prix_camion") and extracted.get("prix_client"):
            prix = extracted

    # Fallback : prix dans cache général utilisateur
    if not prix:
        cached = r.get(f"prix:{user_id}")
        if cached:
            try:
                prix = json.loads(cached)
                logger.info("♻️  Prix récupérés du cache général [%s]", user_id)
            except Exception:
                pass

    logger.info("📦 [%s] Groupe collecté : %d image(s) | prix=%s",
                user_id, len(images_bytes), prix)
    return images_bytes, prix


# ── Fallback OpenAI vision pour images difficiles ─────────────────────────────

def _openai_vision_extract(image_bytes: bytes, idx: int, known_names: list[str]) -> dict:
    """
    Envoie une image directement à GPT-4o vision pour extraction.
    Utilisé quand l OCR ne trouve pas d immatriculation.
    Retourne un dict compatible avec les résultats OCR normaux.
    """
    logger.info("👁️  [image #%d] → OpenAI vision (fallback)", idx)
    t0 = time.perf_counter()

    # Contexte entreprises pour le vision aussi
    ent_context = ""
    if known_names:
        liste = "\n".join(f"  - {n}" for n in known_names)
        ent_context = f"\nEntreprises connues : {liste}\nSi le nom ressemble à l une → utiliser le nom exact."

    system = _VISION_EXTRACT_SYSTEM + ent_context
    b64    = base64.b64encode(image_bytes).decode()

    try:
        with httpx.Client(timeout=OPENAI_TIMEOUT) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=_openai_headers(),
                json={
                    "model": _LLM_MODEL, "temperature": 0, "max_tokens": 200,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                            {"type": "text", "text": system},
                        ],
                    }],
                },
            )
            resp.raise_for_status()
            raw   = resp.json()["choices"][0]["message"]["content"].strip()
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            data  = json.loads(clean)
            logger.info("✅ [image #%d] vision %.2fs → %s", idx, time.perf_counter() - t0, data)
            return {"id": idx, **data, "_source": "vision"}

    except Exception as exc:
        logger.error("❌ [image #%d] vision échoué : %s", idx, exc)
        return {"id": idx, "immatriculation": None, "tonnage_kg": None,
                "entreprise": None, "erreur": str(exc), "_source": "vision"}


# ── Appel OpenAI texte pour un lot ────────────────────────────────────────────

def _call_openai_lot(lot: list[dict], lot_num: int, extract_system: str) -> list[dict]:
    logger.info("🤖 Lot #%d → OpenAI texte (%d reçus)", lot_num, len(lot))
    t0 = time.perf_counter()
    raw_response = ""
    try:
        with httpx.Client(timeout=OPENAI_TIMEOUT) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=_openai_headers(),
                json={
                    "model": _LLM_MODEL, "temperature": 0, "max_tokens": 600,
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
        return [{"id": r["id"], "erreur": f"JSON invalide"} for r in lot]
    except Exception as exc:
        logger.error("❌ Lot #%d erreur : %s", lot_num, exc)
        return [{"id": r["id"], "erreur": str(exc)} for r in lot]


# ── Message récap WhatsApp ────────────────────────────────────────────────────

def _build_recap(results: list[dict], nb_total: int,
                 prix_camion: float, prix_client: float) -> str:
    nb_ok  = sum(1 for r in results if r.get("saved"))
    nb_err = nb_total - nb_ok

    lines = [
        f"✅ *Traitement terminé — {nb_ok}/{nb_total} reçu(s) enregistré(s)*",
        f"💰 Prix camion : {prix_camion} /t  |  💵 Prix client : {prix_client} /t",
        "",
    ]

    for r in sorted(results, key=lambda x: x["id"]):
        num    = r["id"] + 1
        source = " _(vision)_" if r.get("_source") == "vision" else ""
        if r.get("saved"):
            immat  = r.get("immatriculation") or "—"
            kg     = r.get("tonnage_kg")
            tonnes = f"{kg / 1000:.3f} t  ({kg:,.0f} kg)" if kg else "—"
            ent    = r.get("entreprise") or "—"
            lines.append(f"📋 *Reçu #{num}*{source}")
            lines.append(f"   🚛 Immat      : `{immat}`")
            lines.append(f"   ⚖️  Tonnage    : {tonnes}")
            lines.append(f"   🏢 Entreprise : {ent}")
        else:
            err = r.get("erreur") or "Données manquantes"
            lines.append(f"❌ *Reçu #{num}* — non enregistré{source}")
            lines.append(f"   ↳ {err}")
        lines.append("")

    if nb_err:
        lines.append(f"⚠️ {nb_err} reçu(s) ignoré(s) — vérifiez la qualité des images.")

    return "\n".join(lines).strip()


# ── Traitement du groupe complet ──────────────────────────────────────────────

async def _process_group(user_id: str, whatsapp_sender) -> None:
    """
    Appelé après GROUP_WAIT_SEC secondes.
    Collecte toutes les images du groupe, traite, envoie UNE réponse.
    whatsapp_sender est une coroutine async(user_id, message) — injectable.
    """
    logger.info("⏰ [%s] Timer expiré → traitement du groupe", user_id)

    images_bytes, prix = _redis_group_collect(user_id)

    if not images_bytes:
        logger.warning("[%s] Groupe vide après collecte, abandon", user_id)
        return

    nb = len(images_bytes)

    # ── Validation prix ────────────────────────────────────────────────────────
    prix_camion = prix.get("prix_camion") if prix else None
    prix_client = prix.get("prix_client") if prix else None

    if not prix_camion or not prix_client:
        missing = []
        if not prix_camion:
            missing.append("💰 *prix camion* (payé au transporteur par tonne)")
        if not prix_client:
            missing.append("💵 *prix client* (facturé à l entreprise par tonne)")
        msg = (
            "⚠️ *Prix manquants — traitement annulé.*\n\n"
            "Merci de fournir :\n" +
            "\n".join(f"  • {m}" for m in missing) +
            "\n\nExemple : _5000 7000_\nPuis renvoyez vos images."
        )
        await whatsapp_sender(user_id, msg)
        return

    # Cache prix pour prochains webhooks
    try:
        r = get_redis()
        r.setex(f"prix:{user_id}", PRIX_TTL_SEC,
                json.dumps({"prix_camion": prix_camion, "prix_client": prix_client}))
    except Exception:
        pass

    logger.info("💲 [%s] prix_camion=%.0f prix_client=%.0f | %d image(s)",
                user_id, prix_camion, prix_client, nb)

    # ── OCR multithreading ─────────────────────────────────────────────────────
    logger.info("🖼️  OCR sur %d image(s)...", nb)
    ocr_results = extract_text_from_images(images_bytes)

    # ── Tri : images avec immat (OCR OK) vs sans immat (fallback vision) ───────
    ocr_ok:      list[dict]  = []   # {id, ocr_text}
    vision_todo: list[tuple] = []   # (idx, image_bytes)
    failed_ocr:  dict[int, str] = {}

    for idx in sorted(ocr_results):
        entry = ocr_results[idx]
        if entry["error"]:
            logger.warning("⚠️  Image #%d OCR erreur : %s", idx + 1, entry["error"])
            failed_ocr[idx] = entry["error"]
            continue

        text = entry["text"] or ""
        if _has_immat(text):
            # OCR a trouvé une immat → traitement texte normal
            ocr_ok.append({"id": idx, "ocr_text": _clean_ocr(text)})
            logger.info("✅ [image #%d] immat trouvée par OCR", idx + 1)
        else:
            # Pas d immat → fallback vision OpenAI
            vision_todo.append((idx, images_bytes[idx]))
            logger.info("👁️  [image #%d] pas d immat OCR → vision", idx + 1)

    logger.info("📊 OCR OK=%d | Vision=%d | Erreur=%d",
                len(ocr_ok), len(vision_todo), len(failed_ocr))

    # ── Entreprises connues pour fuzzy ─────────────────────────────────────────
    try:
        known_raw   = list_entreprises.invoke({})
        known_names: list[str] = (
            known_raw if isinstance(known_raw, list)
            else [l.strip() for l in str(known_raw).splitlines() if l.strip()]
        )
    except Exception as exc:
        logger.warning("list_entreprises indisponible : %s", exc)
        known_names = []

    # ── Traitement vision (fallback, en série) ─────────────────────────────────
    vision_results: dict[int, dict] = {}
    for idx, img_bytes in vision_todo:
        result = _openai_vision_extract(img_bytes, idx, known_names)
        vision_results[idx] = result

    # ── Traitement OCR OK → lots OpenAI texte en série ────────────────────────
    extract_system  = _build_extract_system(known_names)
    lots = [ocr_ok[i: i + LOT_SIZE] for i in range(0, len(ocr_ok), LOT_SIZE)]
    logger.info("📦 %d lot(s) texte + %d vision", len(lots), len(vision_todo))

    ocr_extracted: list[dict] = []
    for lot_num, lot in enumerate(lots, start=1):
        ocr_extracted.extend(_call_openai_lot(lot, lot_num, extract_system))

    # ── Consolidation + save_trip ──────────────────────────────────────────────
    results_map: dict[int, dict] = {}

    # Erreurs OCR pures
    for idx, err in failed_ocr.items():
        results_map[idx] = {"id": idx, "saved": False, "erreur": f"OCR échoué : {err}"}

    # Fusion résultats OCR texte + vision
    all_items = ocr_extracted + list(vision_results.values())

    for item in all_items:
        idx        = item.get("id", -1)
        is_vision  = item.get("_source") == "vision"
        immat      = item.get("immatriculation")
        tonnage_kg = item.get("tonnage_kg")
        entreprise = item.get("entreprise")

        entry: dict[str, Any] = {
            "id":              idx,
            "saved":           False,
            "immatriculation": immat,
            "tonnage_kg":      tonnage_kg,
            "entreprise":      entreprise,
            "erreur":          item.get("erreur"),
            "_source":         "vision" if is_vision else "ocr",
        }

        if item.get("erreur"):
            results_map[idx] = entry
            continue

        if not tonnage_kg:
            entry["erreur"] = "Tonnage introuvable"
            results_map[idx] = entry
            continue
        if not entreprise:
            entry["erreur"] = "Entreprise introuvable"
            results_map[idx] = entry
            continue

        try:
            save_kwargs: dict[str, Any] = {
                "num_camion":  immat or "INCONNU",
                "entreprise":  entreprise,
                "tonnage":     tonnage_kg,
                "unite":       "kg",
                "prix_camion": prix_camion,
                "prix_client": prix_client,
            }
            logger.info("📤 save_trip #%d → %s", idx + 1, save_kwargs)
            save_trip.invoke(save_kwargs)
            entry["saved"] = True
            logger.info("💾 #%d OK — %s / %s / %.0f kg",
                        idx + 1, immat, entreprise, tonnage_kg)
        except Exception as exc:
            entry["erreur"] = f"Erreur save_trip : {exc}"
            logger.error("❌ save_trip #%d : %s", idx + 1, exc)

        results_map[idx] = entry

    # ── UNE SEULE réponse finale ───────────────────────────────────────────────
    all_results = list(results_map.values())
    recap = _build_recap(all_results, nb_total=nb,
                         prix_camion=prix_camion, prix_client=prix_client)
    logger.info("📤 Récap final [%s] :\n%s", user_id, recap)
    await whatsapp_sender(user_id, recap)


# ── Tâches de groupe en cours (évite les doublons de timers) ─────────────────
_group_tasks: dict[str, asyncio.Task] = {}


async def _schedule_group(user_id: str, whatsapp_sender) -> None:
    """Attend GROUP_WAIT_SEC secondes puis lance le traitement du groupe."""
    await asyncio.sleep(GROUP_WAIT_SEC)
    _group_tasks.pop(user_id, None)
    await _process_group(user_id, whatsapp_sender)


# ── Sender WhatsApp mock (à remplacer par l appel API réel) ──────────────────
# En production, cette fonction envoie via l API WhatsApp Business.
# Ici elle stocke dans Redis pour que l endpoint /poll puisse la récupérer.

async def _whatsapp_send(user_id: str, message: str) -> None:
    """Sauvegarde la réponse dans Redis pour récupération par /poll."""
    try:
        r = get_redis()
        r.lpush(f"outbox:{user_id}", message)
        r.expire(f"outbox:{user_id}", 300)
        logger.info("📬 [%s] Message en outbox Redis", user_id)
    except Exception as exc:
        logger.error("Erreur outbox Redis : %s", exc)


# ── Endpoint principal ────────────────────────────────────────────────────────

@app.post("/agent")
async def agent_endpoint(payload: Payload):
    user_id = payload.user or "default"
    nb_imgs = len(payload.images or ([payload.image] if payload.image else []))
    logger.info("📩 [%s] images=%d  msg=%r", user_id, nb_imgs, payload.message)

    try:
        images_b64 = payload.images or ([payload.image] if payload.image else [])
        images_bytes = _decode_images(payload)

        # ══════════════════════════════════════════════════════════════════════
        # BRANCHE IMAGE
        # ══════════════════════════════════════════════════════════════════════
        if images_bytes:

            # Extraction prix depuis le message (LLM)
            prix: dict | None = None
            if payload.prix_camion and payload.prix_client:
                prix = {"prix_camion": payload.prix_camion,
                        "prix_client": payload.prix_client}
            elif payload.message:
                extracted = _llm_extract_prices(payload.message)
                if extracted.get("prix_camion") and extracted.get("prix_client"):
                    prix = extracted

            # Ajout au buffer groupe Redis
            is_first = _redis_group_add(user_id, images_b64, prix, payload.message or "")

            if is_first:
                # Premier webhook du groupe → ACK + démarrage timer
                ack = (
                    f"⏳ *Traitement en cours...*\n"
                    f"📄 Images reçues — traitement en cours...\n"
                    f"Merci de patienter ✨"
                )
                logger.info("📤 ACK → [%s]", user_id)

                # Annuler timer existant si nécessaire
                if user_id in _group_tasks:
                    _group_tasks[user_id].cancel()

                # Démarrer timer
                task = asyncio.create_task(_schedule_group(user_id, _whatsapp_send))
                _group_tasks[user_id] = task

                return {"ack": ack, "reply": None, "status": "processing"}
            else:
                # Webhook supplémentaire → image ajoutée silencieusement
                logger.info("➕ [%s] Image ajoutée au groupe en attente", user_id)
                return {"status": "queued", "reply": None}

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


# ── Endpoint poll : récupère la réponse finale ────────────────────────────────
# Le webhook WhatsApp appelle GET /poll/{user_id} pour récupérer la réponse
# une fois le traitement terminé, puis l envoie à l utilisateur.

@app.get("/poll/{user_id}")
async def poll_reply(user_id: str):
    """
    Récupère le prochain message en attente dans l outbox Redis.
    Retourne {"reply": "..."} ou {"reply": null} si rien n est prêt.
    """
    try:
        r   = get_redis()
        msg = r.rpop(f"outbox:{user_id}")
        return {"reply": msg, "user": user_id}
    except Exception as exc:
        logger.error("Poll Redis erreur [%s]: %s", user_id, exc)
        return {"reply": None, "user": user_id}
 

# ── Endpoints utilitaires ─────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":          "ok",
        "model":           _LLM_MODEL,
        "lot_size":        LOT_SIZE,
        "openai_timeout":  OPENAI_TIMEOUT,
        "group_wait_sec":  GROUP_WAIT_SEC,
    }


@app.delete("/memory/{user_id}")
async def reset_memory(user_id: str):
    from memory_service import clear_memory
    clear_memory(user_id)
    return {"status": "ok", "message": f"Mémoire effacée pour {user_id}"}