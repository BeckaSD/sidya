"""
memory_service.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mémoire conversationnelle par utilisateur via Redis.

Structure Redis (par user_id) :
  chat:{user_id}:messages  → Liste JSON  (10 derniers messages max)
  chat:{user_id}:summary   → String      (résumé glissant)

Fonctionnement :
  • Chaque échange (user + assistant) est ajouté à la liste
  • Quand la liste dépasse MAX_MESSAGES → on génère un nouveau résumé
    condensé via OpenAI puis on vide la liste
  • Le résumé est toujours injecté en tête du contexte envoyé au LLM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
import redis

logger = logging.getLogger("memory_service")

# ── Config ────────────────────────────────────────────────────────────────────

REDIS_HOST:   str = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT:   int = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB:     int = int(os.getenv("REDIS_DB", "0"))
REDIS_PASS:   str | None = os.getenv("REDIS_PASSWORD", None)

MAX_MESSAGES: int   = 10      # messages conservés avant compression
TTL_SECONDS:  int   = 86400   # expiration clés Redis (24h)

_OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
_LLM_MODEL:      str = os.getenv("LLM_MODEL", "gpt-4o")

# ── Client Redis (singleton) ──────────────────────────────────────────────────

_redis_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host     = REDIS_HOST,
            port     = REDIS_PORT,
            db       = REDIS_DB,
            password = REDIS_PASS,
            decode_responses = True,
        )
    return _redis_client


# ── Clés Redis ────────────────────────────────────────────────────────────────

def _key_messages(user_id: str) -> str:
    return f"chat:{user_id}:messages"

def _key_summary(user_id: str) -> str:
    return f"chat:{user_id}:summary"


# ── Résumé glissant ───────────────────────────────────────────────────────────

def _generate_summary(existing_summary: str, messages: list[dict]) -> str:
    """
    Envoie le résumé existant + les N derniers messages à OpenAI
    pour produire un résumé condensé mis à jour.
    """
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )

    prompt = (
        "Tu es un assistant qui résume des conversations.\n"
        "Voici le résumé actuel de la conversation :\n"
        f"{existing_summary or '(aucun résumé pour l instant)'}\n\n"
        "Voici les nouveaux échanges à intégrer :\n"
        f"{history_text}\n\n"
        "Produis un résumé concis (5 lignes max) en français qui capture "
        "les informations clés : entreprises mentionnées, tonnages, prix, "
        "actions effectuées, demandes en attente."
    )

    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {_OPENAI_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       _LLM_MODEL,
                    "temperature": 0,
                    "max_tokens":  300,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.error("Erreur génération résumé : %s", exc)
        # Fallback : concaténation brute si OpenAI échoue
        return (existing_summary + "\n" + history_text)[-1000:]


# ── API publique ──────────────────────────────────────────────────────────────

def load_context(user_id: str) -> tuple[str, list[dict]]:
    """
    Charge le contexte mémoire d'un utilisateur.

    Retourne :
        (summary: str, messages: list[dict])
        où messages = liste de {role, content}
    """
    r = get_redis()
    try:
        summary_raw  = r.get(_key_summary(user_id)) or ""
        messages_raw = r.lrange(_key_messages(user_id), 0, -1)
        messages     = [json.loads(m) for m in messages_raw]
        return summary_raw, messages
    except Exception as exc:
        logger.error("load_context [%s] : %s", user_id, exc)
        return "", []


def save_exchange(user_id: str, user_msg: str, assistant_msg: str) -> None:
    """
    Enregistre un échange (user + assistant) dans Redis.
    Si la liste dépasse MAX_MESSAGES → compression + reset.
    """
    r = get_redis()
    key_msg = _key_messages(user_id)
    key_sum = _key_summary(user_id)

    try:
        # Ajout des deux nouveaux messages
        for role, content in [("user", user_msg), ("assistant", assistant_msg)]:
            r.rpush(key_msg, json.dumps({"role": role, "content": content}))

        r.expire(key_msg, TTL_SECONDS)
        r.expire(key_sum, TTL_SECONDS)

        # Vérification dépassement
        count = r.llen(key_msg)
        if count >= MAX_MESSAGES:
            logger.info("📦 [%s] %d messages → compression résumé", user_id, count)
            messages_raw = r.lrange(key_msg, 0, -1)
            messages     = [json.loads(m) for m in messages_raw]
            old_summary  = r.get(key_sum) or ""

            new_summary = _generate_summary(old_summary, messages)

            # Atomic : nouveau résumé + liste vidée
            pipe = r.pipeline()
            pipe.set(key_sum, new_summary, ex=TTL_SECONDS)
            pipe.delete(key_msg)
            pipe.execute()

            logger.info("✅ [%s] Résumé mis à jour", user_id)

    except Exception as exc:
        logger.error("save_exchange [%s] : %s", user_id, exc)


def build_memory_prompt(summary: str, messages: list[dict]) -> str:
    """
    Formate le contexte mémoire à injecter dans le prompt système.
    Retourne une chaîne vide si aucun contexte disponible.
    """
    parts: list[str] = []

    if summary:
        parts.append(f"=== RÉSUMÉ DE LA CONVERSATION ===\n{summary}")

    if messages:
        history = "\n".join(
            f"{'Utilisateur' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in messages
        )
        parts.append(f"=== HISTORIQUE RÉCENT ===\n{history}")

    return "\n\n".join(parts)


def clear_memory(user_id: str) -> None:
    """Efface toute la mémoire d'un utilisateur (debug / reset)."""
    r = get_redis()
    try:
        r.delete(_key_messages(user_id), _key_summary(user_id))
        logger.info("🗑️  Mémoire effacée pour [%s]", user_id)
    except Exception as exc:
        logger.error("clear_memory [%s] : %s", user_id, exc)