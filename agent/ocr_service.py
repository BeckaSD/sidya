"""
ocr_service.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Service OCR autonome — indépendant du LLM.
• Une instance RapidOCR par thread (thread-local)
• Multithreading automatique selon le nombre de cœurs disponibles
• Entrée  : liste de bytes (images décodées base64)
• Sortie  : dict  {index: texte_ocr | erreur}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import io
import logging
import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageFile
from rapidocr_onnxruntime import RapidOCR

# ── Configuration ─────────────────────────────────────────────────────────────

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Nombre de cœurs ONNX alloués à chaque instance RapidOCR.
# Exemple : 6 cœurs CPU, 3 threads → 2 cœurs/thread.
CORES_PER_THREAD: int = 2

# Résolution maximale avant redimensionnement (côté le plus long, en pixels).
MAX_SIDE: int = 600

# ── Logging ───────────────────────────────────────────────────────────────────

logger = logging.getLogger("ocr_service")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ── Instance RapidOCR par thread ──────────────────────────────────────────────

_thread_local = threading.local()


def _get_ocr() -> RapidOCR:
    """Retourne (ou crée) l'instance RapidOCR du thread courant."""
    if not hasattr(_thread_local, "ocr"):
        _thread_local.ocr = RapidOCR(
            intra_op_num_threads=CORES_PER_THREAD,
            inter_op_num_threads=CORES_PER_THREAD,
        )
        logger.debug("Nouvelle instance RapidOCR — thread %s", threading.current_thread().name)
    return _thread_local.ocr


# ── Prétraitement image ───────────────────────────────────────────────────────

def _preprocess(image_bytes: bytes) -> np.ndarray:
    """
    Convertit les bytes en image OpenCV niveaux de gris binarisée.
    Redimensionne si le côté max dépasse MAX_SIDE.
    """
    t0 = time.perf_counter()

    # Décodage via PIL (supporte JPEG, PNG, WEBP, etc.)
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    scale = min(MAX_SIDE / max(h, w), 1.0)

    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_w, new_h = w, h

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    logger.debug(
        "Prétraitement: %dx%d → %dx%d (%.2fs)",
        w, h, new_w, new_h, time.perf_counter() - t0,
    )
    return thresh


# ── Traitement d'une image ────────────────────────────────────────────────────

def _process_single(index: int, image_bytes: bytes) -> tuple[int, str | None, str | None]:
    """
    Traite une image et retourne (index, texte_ocr, erreur).
    Appelé dans un thread worker.
    """
    thread = threading.current_thread().name
    t0 = time.perf_counter()
    logger.info("▶ [%s] image #%d", thread, index)

    try:
        ocr = _get_ocr()
        preprocessed = _preprocess(image_bytes)

        t_ocr = time.perf_counter()
        result, _ = ocr(preprocessed)
        logger.info("🔍 [image #%d] OCR: %.2fs", index, time.perf_counter() - t_ocr)

        if not result:
            logger.warning("⚠ [image #%d] Aucun texte détecté", index)
            return index, "", None  # texte vide, pas d'erreur

        # Filtrer les fragments trop courts (bruit)
        texte = "\n".join(
            ligne[1] for ligne in result if len(ligne[1].strip()) > 2
        )

        logger.info("✅ [image #%d] total: %.2fs", index, time.perf_counter() - t0)
        return index, texte, None

    except Exception as exc:  # noqa: BLE001
        logger.exception("❌ [image #%d] erreur: %s", index, exc)
        return index, None, str(exc)


# ── Point d'entrée public ─────────────────────────────────────────────────────

def extract_text_from_images(images_bytes: list[bytes]) -> dict[int, Any]:
    """
    Extrait le texte OCR de plusieurs images en parallèle.

    Paramètres
    ----------
    images_bytes : list[bytes]
        Liste d'images encodées en bytes (déjà décodées depuis base64).

    Retourne
    --------
    dict[int, Any]
        {
            0: {"text": "...", "error": None},
            1: {"text": None, "error": "message d'erreur"},
            ...
        }
    """
    nb = len(images_bytes)
    if nb == 0:
        return {}

    total_cores = multiprocessing.cpu_count()
    workers = max(1, min(nb, total_cores // CORES_PER_THREAD))

    logger.info(
        "🚀 %d image(s) | %d thread(s) | %d cœurs | %d cœurs/thread",
        nb, workers, total_cores, CORES_PER_THREAD,
    )

    t_global = time.perf_counter()
    results: dict[int, Any] = {}

    if nb == 1:
        # Pas de surcoût threadpool pour une seule image
        idx, text, err = _process_single(0, images_bytes[0])
        results[idx] = {"text": text, "error": err}
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_single, i, img_bytes): i
                for i, img_bytes in enumerate(images_bytes)
            }
            for future in as_completed(futures):
                idx, text, err = future.result()
                results[idx] = {"text": text, "error": err}

    elapsed = time.perf_counter() - t_global
    logger.info("📊 OCR terminé en %.2fs pour %d image(s)", elapsed, nb)
    return results