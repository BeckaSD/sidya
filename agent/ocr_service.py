"""
ocr_service.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Service OCR autonome — indépendant du LLM.

Prétraitement multi-pipeline (cascade, stop dès immat trouvée) :
  1) Otsu + CLAHE
  2) Adaptatif gaussien + unsharp mask
  3) Bilatéral + seuil fixe
  4) Couleur brut (sans binarisation)

Multithreading :
  • Une instance RapidOCR par thread (thread-local)
  • Workers = cpu_count // CORES_PER_THREAD

Interface publique inchangée :
  extract_text_from_images(list[bytes]) → dict[int, {text, error}]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import io
import logging
import multiprocessing
import os
import re
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

# Limite les threads ONNX par instance RapidOCR
CORES_PER_THREAD: int = 2

# Résolution cible pour l OCR (côté le plus long, en pixels)
TARGET_SIZE: int = 1600

# Pattern immatriculation mauritanienne : 4 chiffres + 2 lettres + 2 chiffres
_IMMAT_PATTERN = re.compile(r'\b(\d{4}[A-Z]{2}\d{2})\b', re.IGNORECASE)

# Limite les threads systèmes BLAS/OpenMP pour éviter la contention
os.environ.setdefault("OMP_NUM_THREADS",     "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ.setdefault("MKL_NUM_THREADS",     "2")

# ── Logging ───────────────────────────────────────────────────────────────────

logger = logging.getLogger("ocr_service")

# ── Instance RapidOCR par thread ──────────────────────────────────────────────

_thread_local = threading.local()


def _get_ocr() -> RapidOCR:
    """Retourne (ou crée) l instance RapidOCR du thread courant."""
    if not hasattr(_thread_local, "ocr"):
        _thread_local.ocr = RapidOCR(
            intra_op_num_threads=CORES_PER_THREAD,
            inter_op_num_threads=CORES_PER_THREAD,
        )
        logger.debug("Nouvelle instance RapidOCR — thread %s", threading.current_thread().name)
    return _thread_local.ocr


# ── Helpers image ─────────────────────────────────────────────────────────────

def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """Décode bytes → image BGR OpenCV (supporte JPEG, PNG, WEBP, etc.)."""
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _detect_and_crop(img: np.ndarray) -> np.ndarray:
    """
    Détecte le document (zone blanche principale) et le recadre.
    Retourne l image originale si aucun contour pertinent trouvé.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    img_area = img.shape[0] * img.shape[1]
    best, best_area = None, 0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area  = w * h
        ratio = w / max(h, 1)
        if area > img_area * 0.15 and area > best_area and 0.3 < ratio < 4.0:
            best, best_area = (x, y, w, h), area

    if best:
        x, y, w, h = best
        pad = 20
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img.shape[1] - x, w + pad * 2)
        h = min(img.shape[0] - y, h + pad * 2)
        return img[y:y+h, x:x+w]

    return img


def _resize(img: np.ndarray, target: int = TARGET_SIZE) -> np.ndarray:
    """Redimensionne l image si le côté max dépasse target."""
    h, w = img.shape[:2]
    scale = target / max(h, w)
    if abs(scale - 1.0) > 0.05:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=interp)
    return img


# ── Pipelines de prétraitement ────────────────────────────────────────────────

def _pipeline_otsu_clahe(img: np.ndarray) -> np.ndarray:
    """Pipeline 1 : niveaux de gris + débruitage + CLAHE + Otsu."""
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    gray  = cv2.fastNlMeansDenoising(gray, h=10)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _pipeline_adaptive(img: np.ndarray) -> np.ndarray:
    """Pipeline 2 : débruitage + CLAHE renforcé + unsharp mask + seuil adaptatif."""
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray  = cv2.fastNlMeansDenoising(gray, h=15)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)
    blur  = cv2.GaussianBlur(gray, (0, 0), 3)
    gray  = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 25, 8,
    )


def _pipeline_bilateral(img: np.ndarray) -> np.ndarray:
    """Pipeline 3 : filtre bilatéral + seuil fixe."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh


def _pipeline_raw_color(img: np.ndarray) -> np.ndarray:
    """Pipeline 4 : couleur brut redimensionné — fallback si tous échouent."""
    return _resize(img, TARGET_SIZE)


# Ordre d exécution — stop dès qu une immatriculation est trouvée
_PIPELINES: list[tuple[str, Any]] = [
    ("Otsu+CLAHE",   _pipeline_otsu_clahe),
    ("Adaptatif",    _pipeline_adaptive),
    ("Bilatéral",    _pipeline_bilateral),
    ("Couleur brut", _pipeline_raw_color),
]


# ── Extraction texte + immat ──────────────────────────────────────────────────

def _run_ocr(ocr: RapidOCR, img: np.ndarray) -> str:
    """Lance RapidOCR sur une image prétraitée et retourne le texte filtré."""
    result, _ = ocr(img)
    if not result:
        return ""
    return "\n".join(r[1] for r in result if len(r[1].strip()) > 2)


def _find_immat(text: str) -> str | None:
    """Cherche une immatriculation mauritanienne dans le texte OCR."""
    match = _IMMAT_PATTERN.search(text.upper())
    return match.group(1).upper() if match else None


# ── Traitement d une image (cascade multi-pipeline) ──────────────────────────

def _process_single(index: int, image_bytes: bytes) -> tuple[int, str | None, str | None]:
    """
    Traite une image via la cascade de pipelines et retourne :
        (index, meilleur_texte | None, erreur | None)

    Stratégie :
    - Essaie chaque pipeline dans l ordre
    - Conserve le texte du premier pipeline qui trouve une immatriculation
    - Si aucun pipeline ne trouve d immat, retourne le texte le plus long
    - Stop dès qu une immat valide est trouvée
    """
    t0 = time.perf_counter()
    logger.info("▶ [image #%d] début traitement", index)

    try:
        ocr       = _get_ocr()
        img_bgr   = _bytes_to_bgr(image_bytes)
        img_crop  = _detect_and_crop(img_bgr)
        img_ready = _resize(img_crop, TARGET_SIZE)

        best_text:     str       = ""
        best_pipeline: str       = "aucun"
        best_immat:    str|None  = None

        for pipe_name, pipe_fn in _PIPELINES:
            try:
                t_pipe = time.perf_counter()
                img_proc = pipe_fn(img_ready)
                text     = _run_ocr(ocr, img_proc)
                immat    = _find_immat(text)

                logger.info(
                    "   🔬 [image #%d] %s → %d chars | immat=%s (%.2fs)",
                    index, pipe_name, len(text), immat or "❌",
                    time.perf_counter() - t_pipe,
                )

                # Premier pipeline avec immat → meilleur résultat
                if immat and not best_immat:
                    best_text     = text
                    best_immat    = immat
                    best_pipeline = pipe_name

                # Garde le texte le plus long comme fallback
                if not best_immat and len(text) > len(best_text):
                    best_text     = text
                    best_pipeline = pipe_name

                # Stop dès qu on a une immatriculation valide
                if best_immat:
                    break

            except Exception as exc:
                logger.warning("   ⚠️  [image #%d] pipeline %s : %s", index, pipe_name, exc)

        elapsed = time.perf_counter() - t0
        logger.info(
            "✅ [image #%d] pipeline=%s | immat=%s | %.2fs",
            index, best_pipeline, best_immat or "NON TROUVÉE", elapsed,
        )

        if not best_text:
            logger.warning("⚠ [image #%d] Aucun texte détecté", index)

        return index, best_text or "", None

    except Exception as exc:
        logger.exception("❌ [image #%d] erreur : %s", index, exc)
        return index, None, str(exc)


# ── Point d entrée public (interface inchangée) ───────────────────────────────

def extract_text_from_images(images_bytes: list[bytes]) -> dict[int, Any]:
    """
    Extrait le texte OCR de plusieurs images en parallèle.

    Paramètres
    ----------
    images_bytes : list[bytes]
        Liste d images encodées en bytes (déjà décodées depuis base64).

    Retourne
    --------
    dict[int, Any]
        {
            0: {"text": "...", "error": None},
            1: {"text": None, "error": "message d erreur"},
            ...
        }
    """
    nb = len(images_bytes)
    if nb == 0:
        return {}

    total_cores = multiprocessing.cpu_count()
    workers     = max(1, min(nb, total_cores // CORES_PER_THREAD))

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