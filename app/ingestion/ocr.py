"""
app/ingestion/ocr.py — OCR for scanned PDFs and images using Tesseract.

If Tesseract confidence is very low, falls back to LLM vision (gpt-4o style)
via OpenRouter if a vision-capable model is available.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def _tesseract_image(image) -> tuple[str, float]:
    """Run pytesseract on a PIL Image. Returns (text, mean_confidence)."""
    import pytesseract  # type: ignore

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    texts = [t for t, c in zip(data["text"], data["conf"]) if int(c) > 0]
    confs = [int(c) for c in data["conf"] if int(c) > 0]
    text = " ".join(texts)
    confidence = sum(confs) / max(len(confs), 1)
    return text, confidence


def ocr_image_path(image_path: Path) -> str:
    """
    OCR a single image file.  Returns extracted text.
    Falls back to LLM vision if Tesseract confidence < 40.
    """
    from PIL import Image  # type: ignore

    img = Image.open(image_path)
    text, confidence = _tesseract_image(img)
    log.debug("Tesseract confidence=%.1f for %s", confidence, image_path.name)

    if confidence < 40:
        log.info("Low OCR confidence (%.1f); trying LLM vision fallback", confidence)
        llm_text = _llm_vision_fallback(image_path)
        if llm_text:
            return llm_text

    return text


def ocr_pdf_pages(pdf_path: Path, dpi: int = 200) -> str:
    """
    Convert each page of a PDF to an image and OCR it.
    Used when PyMuPDF extracts < 100 chars/page (scanned PDF).
    """
    try:
        from pdf2image import convert_from_path  # type: ignore
    except ImportError:
        log.warning("pdf2image not installed; OCR will be skipped")
        return ""

    pages = convert_from_path(str(pdf_path), dpi=dpi)
    all_text: list[str] = []
    for i, page in enumerate(pages):
        text, conf = _tesseract_image(page)
        if conf < 40:
            log.info("Page %d: low confidence (%.1f), trying vision fallback", i + 1, conf)
            # Save page as temp PNG for vision
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            page.save(tmp_path)
            llm_text = _llm_vision_fallback(tmp_path)
            tmp_path.unlink(missing_ok=True)
            all_text.append(llm_text or text)
        else:
            all_text.append(text)
    return "\n\n".join(all_text)


def _llm_vision_fallback(image_path: Path) -> Optional[str]:
    """
    Send the image to a vision-capable LLM via OpenRouter as a base64 payload.
    Returns None on failure so the caller can use Tesseract output instead.
    """
    try:
        from app.config import get_settings
        from openai import OpenAI  # type: ignore

        cfg = get_settings()
        client = OpenAI(
            api_key=cfg.openrouter_api_key,
            base_url=cfg.openrouter_base_url,
        )

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        ext = image_path.suffix.lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/png")

        resp = client.chat.completions.create(
            model=cfg.llm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": "Extract all text from this image verbatim. Return only the extracted text, no commentary.",
                        },
                    ],
                }
            ],
            max_tokens=2048,
        )
        return resp.choices[0].message.content
    except Exception as e:
        log.warning("LLM vision fallback failed: %s", e)
        return None
