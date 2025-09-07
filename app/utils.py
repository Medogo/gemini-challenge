import io, json
from typing import Any, Dict, List, Optional
from PIL import Image


def parse_json_safely(text: str) -> Any:
    """
    Essaie de parser du JSON même si le modèle a ajouté du texte parasite.
    """
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text, "warning": "JSON not strictly valid; returning raw text."}


def load_image_from_bytes(data: bytes) -> Optional[Image.Image]:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


# ---------- Sanitisers (blindage contre champs null/absents) ----------

def sanitize_variants_dict(d: dict) -> dict:
    out = {"variants": []}
    for v in d.get("variants", []) if isinstance(d, dict) else []:
        out["variants"].append({
            "channel": (v.get("channel") or "Unknown"),
            "title": (v.get("title") or ""),
            "body": (v.get("body") or ""),
            "hashtags": v.get("hashtags") or [],
            "cta": (v.get("cta") or "")
        })
    return out


def sanitize_schedule_dict(d: dict) -> dict:
    out = {"schedule": []}
    for ch in d.get("schedule", []) if isinstance(d, dict) else []:
        slots = []
        for s in ch.get("slots", []) or []:
            slots.append({
                "iso": (s.get("iso") or ""),
                "why": (s.get("why") or "")
            })
        out["schedule"].append({
            "channel": (ch.get("channel") or ""),
            "slots": slots
        })
    return out


def sanitize_image_prompt_dict(d: dict) -> dict:
    if not isinstance(d, dict):
        return {"image_prompt": ""}
    return {"image_prompt": d.get("image_prompt") or ""}
