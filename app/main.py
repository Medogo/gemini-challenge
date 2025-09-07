# app/main.py
import json
import re
import os
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Any
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import google.generativeai as genai
import speech_recognition as sr
from pydub import AudioSegment
from pydub.utils import which
from PIL import Image, ImageDraw, ImageFont

from app.config import settings
from app.schemas import (
    Channel, GenerateRequest, GenerateResponse,
    Variants, ScheduleResponse, ImagePrompt,
)
from app.utils import (
    parse_json_safely, load_image_from_bytes,
    sanitize_variants_dict, sanitize_schedule_dict, sanitize_image_prompt_dict,
)
from app.services.packaging import build_zip_kit
from app.services.exporters import build_ics, build_csv


# ======================
# Constantes & limites
# ======================
IMAGE_MAX_BYTES = 20 * 1024 * 1024      # 20 Mo (image + style_ref)
MEDIA_MAX_BYTES = 100 * 1024 * 1024     # 100 Mo (audio/vidéo)
PLACEHOLDER_FONT = None  # laisser None -> police par défaut PIL


# ----------------------
# pydub / ffmpeg (local)
# ----------------------
FFMPEG = which("ffmpeg") or "/opt/homebrew/bin/ffmpeg" or "/usr/local/bin/ffmpeg"
FFPROBE = which("ffprobe") or "/opt/homebrew/bin/ffprobe" or "/usr/local/bin/ffprobe"
if FFMPEG:
    AudioSegment.converter = FFMPEG
    AudioSegment.ffmpeg = FFMPEG
if FFPROBE:
    AudioSegment.ffprobe = FFPROBE

FFMPEG_MISSING = not bool(FFMPEG and FFPROBE)



# ----------------------
# FastAPI setup
# ----------------------
app = FastAPI(title="Social Creative Coach (Gemini)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ----------------------
# Gemini setup (texte)
# ----------------------
if not settings.GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY manquante. Renseigne-la dans l'env ou .env")
genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel(settings.GEMINI_MODEL)

# Nom du modèle image (ex: "gemini-2.5-flash-image-preview")
GEMINI_IMAGE_MODEL = getattr(settings, "GEMINI_IMAGE_MODEL", None)


# ----------------------
# Helpers audio / vidéo
# ----------------------
def _guess_media_format(filename: str | None, content_type: str | None) -> Optional[str]:
    """
    Devine le format pour pydub/ffmpeg.
    Gère audio ET vidéo (mp4, webm, mkv, mov).
    """
    audio_exts = {"webm", "mp3", "m4a", "ogg", "wav", "aac"}
    video_exts = {"mp4", "webm", "mkv", "mov"}

    if content_type:
        ct = content_type.lower()
        if ct.startswith("audio/"):
            subtype = ct.split("/", 1)[1]
            if subtype == "mpeg": return "mp3"
            if subtype in {"x-m4a"}: return "m4a"
            if subtype in audio_exts: return subtype
        if ct.startswith("video/"):
            if "mp4" in ct: return "mp4"
            if "webm" in ct: return "webm"
            if "ogg" in ct: return "ogg"
            if "mkv" in ct or "x-matroska" in ct: return "mkv"
            if "quicktime" in ct or "mov" in ct: return "mov"

    if filename and "." in filename:
        ext = filename.lower().rsplit(".", 1)[-1]
        if ext in audio_exts or ext in video_exts:
            return ext
    return None


def convert_media_to_wav_excerpt_mono(media_bytes: bytes, src_format: Optional[str], max_seconds: int = 60) -> bytes:
    """
    Charge l'audio/vidéo via ffmpeg, coupe aux 60s, normalise en mono 16 kHz WAV.
    """
    seg = AudioSegment.from_file(BytesIO(media_bytes), format=src_format if src_format else None)
    if len(seg) > max_seconds * 1000:
        seg = seg[: max_seconds * 1000]
    seg = seg.set_channels(1).set_frame_rate(16000)
    out = BytesIO()
    seg.export(out, format="wav")
    return out.getvalue()


def speech_to_text_wav_bytes(wav_bytes: bytes, language: str = "fr-FR") -> Optional[str]:
    """
    Transcrit la piste WAV (≤ ~60s). Si ça échoue, renvoie None.
    """
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(BytesIO(wav_bytes)) as source:
            audio_data = recognizer.record(source)  # 60s déjà découpées
        return recognizer.recognize_google(audio_data, language=language)
    except Exception:
        return None


# ----------------------
# Règles par canal (guidage + post-processing)
# ----------------------
CHANNEL_RULES = {
    "LinkedIn":  "- 3–6 lignes, ton pro, 3–5 hashtags, pas d’abus d’emojis.",
    "Instagram": "- 1–2 lignes punchy, 5–10 hashtags pertinents, émojis permis.",
    "X":         "- Très concis (<280), 1–3 hashtags, CTA implicite fort.",
    "Facebook":  "- Ton convivial, 2–4 lignes, 3–5 hashtags maximum.",
    "TikTok":    "- Accroche très courte, appel à l’action créatif, 3–6 hashtags.",
}

SOFT_LIMITS = {
    "LinkedIn":  {"body_chars": 900,  "hashtags_max": 5,  "title_chars": 220},
    "Instagram": {"body_chars": 400,  "hashtags_max": 10, "title_chars": 220},
    "X":         {"body_chars": 270,  "hashtags_max": 3,  "title_chars": 120},
    "Facebook":  {"body_chars": 500,  "hashtags_max": 5,  "title_chars": 220},
    "TikTok":    {"body_chars": 180,  "hashtags_max": 6,  "title_chars": 100},
}


def _truncate(text: str, n: int) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    return text if len(text) <= n else text[: max(0, n-1)] + "…"


def normalize_variants_for_channels(variants_dict: dict) -> dict:
    cleaned = {"variants": []}
    for v in variants_dict.get("variants", []):
        ch = v.get("channel") or "Unknown"
        title = v.get("title") or ""
        body = v.get("body") or ""
        tags = v.get("hashtags") or []
        cta = v.get("cta") or ""

        limits = SOFT_LIMITS.get(ch, {"body_chars": 500, "hashtags_max": 5, "title_chars": 220})

        title = _truncate(title, limits["title_chars"])
        body  = _truncate(body,  limits["body_chars"])

        if ch == "X":
            combo = (title + " " + body).strip()
            body = _truncate(combo, limits["body_chars"])
            title = ""

        norm = []
        seen = set()
        for t in tags:
            t = str(t).strip().lstrip("#")
            if not t:
                continue
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            norm.append(t)
            if len(norm) >= limits["hashtags_max"]:
                break

        cleaned["variants"].append({
            "channel": ch,
            "title": title,
            "body": body,
            "hashtags": norm,
            "cta": cta
        })
    return cleaned


# ----------------------
# Prompts builders
# ----------------------
def build_variants_prompt(req: GenerateRequest, channel_rules: dict, image_was_used: bool) -> str:
    rules = "\n".join(
        [f"* {ch}: {desc}" for ch, desc in channel_rules.items() if ch in [c.value for c in req.channels]]
    )
    forbidden = ", ".join(req.forbid) if req.forbid else "aucune"
    audience = req.target_audience or "grand public"
    cta_hint = req.call_to_action_hint or "Inciter à cliquer / acheter / s'inscrire (selon contexte)."
    brand_name = req.brand_name or "la marque"
    brand_color = req.brand_color or "couleur dominante de la marque"

    return f"""
Langue de sortie: {req.language}
Tu es un coach créatif senior en social media.
Brief: {req.brief}
Audience cible: {audience}
Canaux: {[c.value for c in req.channels]}
Ton de marque: {req.brand_tone}
Brand cues: nom = {brand_name}, couleur dominante = {brand_color}
Contraintes par canal:
{rules}

Mots/mentions à éviter: {forbidden}
Indice de CTA: {cta_hint}
Image/Media fourni utilisé pour contexte: {str(image_was_used).lower()}

Objectif: génère 3 variantes **par canal**.

Format de réponse: JSON valide, compact, **STRICTEMENT sans texte avant/après**:
{{
  "variants": [
    {{"channel":"LinkedIn","title":"...","body":"...","hashtags":["..."],"cta":"..."}},
    ...
  ]
}}

Rappels de qualité:
- Pas de phrase fourre-tout, pas de verbosité.
- Hashtags ciblés (pas génériques type #love).
- Respecter les contraintes de longueur par canal.
- Aucun champ ne doit être null : utiliser "" pour title/body/cta et [] pour hashtags.
""".strip()


def build_schedule_prompt(req: GenerateRequest) -> str:
    return f"""
Langue de sortie: {req.language}
Propose 2 créneaux de publication **par canal** sur les 7 prochains jours (timezone: {req.timezone}),
avec une justification brève "why" (1 phrase).
Réponds en JSON **STRICTEMENT** (aucun texte avant/après):
{{
  "schedule": [
    {{"channel":"LinkedIn","slots":[{{"iso":"YYYY-MM-DDTHH:MM","why":"..."}}]}},
    ...
  ]
}}
Contexte brief: {req.brief}
Canaux: {[c.value for c in req.channels]}
""".strip()


def build_image_prompt_prompt(req: GenerateRequest) -> str:
    brand_name = req.brand_name or "la marque"
    brand_color = req.brand_color or ""
    brand_line = f"Nom de marque: {brand_name}." if brand_name else ""
    color_line = f"Couleur dominante de la marque: {brand_color}." if brand_color else ""

    return f"""
Langue de sortie: {req.language}
Décris un **prompt d'image** publicitaire **1080x1080** très précis pour ce brief:
{req.brief}

Inclure: style visuel, lumière, composition, arrière-plan, ambiance, cadrage, typographie possible,
{brand_line} {color_line}
mots-clés artistiques utiles.
Réponds en JSON **STRICTEMENT** (aucun texte avant/après):
{{"image_prompt":"..."}}
""".strip()


def build_image_analysis_prompt(language: str = "fr") -> str:
    return f"""
Langue de sortie: {language}
Analyse l'image fournie et renvoie un JSON **STRICTEMENT** sans texte autour:
{{
  "image_summary": {{
    "caption": "...",
    "objects": ["...", "..."],
    "colors": ["...", "..."],
    "brand_cues": "...",
    "style": "...",
    "product": "...",
    "mood": "..."
  }}
}}
""".strip()


def build_markdown_export(project_name: str, gen_resp: GenerateResponse) -> bytes:
    """Crée un markdown lisible (posts + planning)."""
    v = gen_resp.variants.model_dump()
    s = gen_resp.schedule.model_dump()
    ip = gen_resp.image_prompt.model_dump()

    lines = [f"# {project_name}", ""]
    lines += ["## Prompt d’image", "", "```", ip.get("image_prompt", ""), "```", ""]
    lines += ["## Variantes", ""]
    for item in v.get("variants", []):
        lines += [
            f"### {item.get('channel','')}",
            f"**Titre** : {item.get('title','')}",
            "",
            item.get("body",""),
            "",
            f"**Hashtags** : " + " ".join(f"#{t}" for t in item.get("hashtags", [])),
            f"**CTA** : {item.get('cta','')}",
            "",
        ]
    lines += ["## Planning (7j)", ""]
    for ch in s.get("schedule", []):
        lines += [f"### {ch.get('channel','')}", ""]
        for slot in ch.get("slots", []):
            lines += [f"- {slot.get('iso','')} — {slot.get('why','')}", ""]
    return ("\n".join(lines)).encode("utf-8")


# ----------------------
# Routes
# ----------------------
@app.get("/health")
def health():
    return {"status": "ok", "model": settings.GEMINI_MODEL, "image_model": GEMINI_IMAGE_MODEL or "placeholder"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    brief: str = Form("", description="Description de la campagne / idée (peut être vide si image OU audio/vidéo joint)"),
    channels: str = Form('["LinkedIn","Instagram","X"]', description="JSON array of channels"),
    brand_tone: str = Form("professionnel, clair, orienté valeur"),
    language: str = Form("fr"),
    target_audience: str = Form(""),
    call_to_action_hint: str = Form(""),
    forbid: str = Form("[]"),
    timezone: str = Form("Africa/Porto-Novo"),
    brand_name: str = Form(""),
    brand_color: str = Form(""),
    image: UploadFile | None = File(default=None),
    audio: UploadFile | None = File(default=None),
):
    """
    Génère:
      - variants (posts par canal),
      - schedule (créneaux),
      - image_prompt (prompt 1080x1080).
    Supporte image et audio/vidéo en entrée (optionnels).
    """
    # 0) Garde-fou initial : au moins un signal (brief, image ou audio/vidéo)
    if not brief.strip() and image is None and audio is None:
        raise HTTPException(status_code=400, detail="Fournis au moins un brief, une image ou un audio/vidéo.")

    # 1) Parse JSON des champs texte
    try:
        channels_list = json.loads(channels)
        forbid_list = json.loads(forbid)
    except Exception:
        raise HTTPException(status_code=400, detail="`channels` et/ou `forbid` doivent être du JSON valide.")

    # Convertir strings -> Enum Channel (validation)
    try:
        channels_enum: List[Channel] = [Channel(c) for c in channels_list]
    except Exception:
        raise HTTPException(status_code=400, detail="`channels` contient un canal inconnu.")

    # 2) Construire la requête structurée
    req = GenerateRequest(
        brief=brief,
        channels=channels_enum,
        brand_tone=brand_tone,
        language=language,
        target_audience=target_audience or None,
        call_to_action_hint=call_to_action_hint or None,
        forbid=forbid_list,
        timezone=timezone,
        brand_name=brand_name or None,
        brand_color=brand_color or None,
    )

    # 3) Média (audio/vidéo) -> transcript (facultatif)
    transcript_text: Optional[str] = None
    if audio is not None:
        if FFMPEG_MISSING:
            raise HTTPException(
                status_code=500,
                detail="ffmpeg manquant sur le serveur. Installez ffmpeg/ffprobe pour traiter audio/vidéo."
            )
        media_bytes = await audio.read()
        if len(media_bytes) > MEDIA_MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Fichier audio/vidéo trop lourd (> {MEDIA_MAX_BYTES // (1024*1024)} Mo)."
            )
        src_format = _guess_media_format(audio.filename, audio.content_type)
        try:
            wav_excerpt = convert_media_to_wav_excerpt_mono(media_bytes, src_format=src_format, max_seconds=60)
        except Exception:
            raise HTTPException(status_code=400, detail="Impossible de décoder le média (ffmpeg/pydub).")

        lang_code = "fr-FR" if (req.language or "fr").lower().startswith("fr") else "en-US"
        transcript_text = speech_to_text_wav_bytes(wav_excerpt, language=lang_code)

        if transcript_text:
            if not req.brief.strip():
                req.brief = transcript_text
            else:
                req.brief = f"{req.brief}\n\n[Transcription 60s]: {transcript_text}".strip()

    # 4) Image -> contexte (facultatif)
    image_used = False
    parts = []
    if image is not None:
        img_bytes = await image.read()
        if len(img_bytes) > IMAGE_MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Image trop lourde (> {IMAGE_MAX_BYTES // (1024*1024)} Mo)."
            )
        pil = load_image_from_bytes(img_bytes)
        if pil:
            image_used = True
            parts.append(pil)  # SDK accepte PIL.Image

    # 4-bis) Brief auto depuis image si brief toujours vide
    image_analysis_obj = None
    if not req.brief.strip():
        if image_used:
            try:
                analysis_prompt = build_image_analysis_prompt(req.language)
                resp_analysis = model.generate_content(parts + [analysis_prompt])
                analysis_raw = parse_json_safely(resp_analysis.text or "")
                image_summary = analysis_raw.get("image_summary", {}) if isinstance(analysis_raw, dict) else {}
                derived_brief = (
                    f"Image: {image_summary.get('caption','')}. "
                    f"Produit: {image_summary.get('product','')}. "
                    f"Style: {image_summary.get('style','')}. "
                    f"Couleurs: {', '.join(image_summary.get('colors', []))}. "
                    f"Ambiance: {image_summary.get('mood','')}."
                ).strip()
                if derived_brief:
                    req.brief = derived_brief
                    image_analysis_obj = image_summary
            except Exception:
                req.brief = "Générer des posts basés uniquement sur le contenu visuel de l'image fournie."

    # 4-ter) Revalidation finale (au moins un signal exploitable)
    if not req.brief.strip() and not transcript_text and not image_used:
        raise HTTPException(
            status_code=400,
            detail="Brief vide, image absente et audio/vidéo inaudible. Fournis au moins un brief, une image ou un média compréhensible."
        )

    # ----------------------
    # Appels Gemini (texte)
    # ----------------------
    # Variants
    try:
        variants_prompt = build_variants_prompt(req, CHANNEL_RULES, image_used)
        resp_variants = model.generate_content(parts + [variants_prompt]) if parts else model.generate_content(variants_prompt)
        variants_obj_raw = parse_json_safely(resp_variants.text or "")
        variants_obj = sanitize_variants_dict(variants_obj_raw)
        variants_obj = normalize_variants_for_channels(variants_obj)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini variants error: {e}")

    # Schedule
    try:
        schedule_prompt = build_schedule_prompt(req)
        resp_schedule = model.generate_content(schedule_prompt)
        schedule_obj_raw = parse_json_safely(resp_schedule.text or "")
        schedule_obj = sanitize_schedule_dict(schedule_obj_raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini schedule error: {e}")

    # Image prompt
    try:
        img_prompt = build_image_prompt_prompt(req)
        resp_img = model.generate_content(img_prompt)
        image_prompt_obj_raw = parse_json_safely(resp_img.text or "")
        image_prompt_obj = sanitize_image_prompt_dict(image_prompt_obj_raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini image prompt error: {e}")

    # ----------------------
    # Réponse Pydantic sûre
    # ----------------------
    return GenerateResponse(
        variants=Variants(**variants_obj),
        schedule=ScheduleResponse(**schedule_obj),
        image_prompt=ImagePrompt(**image_prompt_obj),
        input_image_used=image_used,
        input_audio_transcript=transcript_text,
        image_analysis=image_analysis_obj,
    )


@app.post("/kit")
async def kit_zip(
    project_name: str = Form("Social Creative Coach"),
    brief: str = Form(""),
    channels: str = Form('["LinkedIn","Instagram","X"]'),
    brand_tone: str = Form("professionnel, clair, orienté valeur"),
    language: str = Form("fr"),
    target_audience: str = Form(""),
    call_to_action_hint: str = Form(""),
    forbid: str = Form("[]"),
    timezone: str = Form("Africa/Porto-Novo"),
    brand_name: str = Form(""),
    brand_color: str = Form(""),
    image: UploadFile | None = File(default=None),
    audio: UploadFile | None = File(default=None),
):
    """
    Génère puis renvoie un ZIP (variants + schedule + image_prompt + .txt par canal).
    """
    gen_resp: GenerateResponse = await generate(
        brief=brief,
        channels=channels,
        brand_tone=brand_tone,
        language=language,
        target_audience=target_audience,
        call_to_action_hint=call_to_action_hint,
        forbid=forbid,
        timezone=timezone,
        brand_name=brand_name,
        brand_color=brand_color,
        image=image,
        audio=audio,
    )

    zip_bytes = build_zip_kit(
        project_name=project_name,
        variants_json=gen_resp.variants.model_dump(),
        schedule_json=gen_resp.schedule.model_dump(),
        image_prompt_json=gen_resp.image_prompt.model_dump(),
    )
    headers = {"Content-Disposition": f'attachment; filename="{project_name.replace(" ", "_")}_kit.zip"'}
    return StreamingResponse(BytesIO(zip_bytes), media_type="application/zip", headers=headers)


@app.post("/export/ics")
async def export_ics(
    project_name: str = Form("Social Creative Coach"),
    brief: str = Form(""),
    channels: str = Form('["LinkedIn","Instagram","X"]'),
    brand_tone: str = Form("professionnel, clair, orienté valeur"),
    language: str = Form("fr"),
    target_audience: str = Form(""),
    call_to_action_hint: str = Form(""),
    forbid: str = Form("[]"),
    timezone: str = Form("Africa/Porto-Novo"),
    brand_name: str = Form(""),
    brand_color: str = Form(""),
    image: UploadFile | None = File(default=None),
    audio: UploadFile | None = File(default=None),
):
    """
    Génère le planning puis renvoie un fichier .ics (1 VEVENT par créneau).
    """
    gen_resp: GenerateResponse = await generate(
        brief=brief, channels=channels, brand_tone=brand_tone, language=language,
        target_audience=target_audience, call_to_action_hint=call_to_action_hint,
        forbid=forbid, timezone=timezone, brand_name=brand_name, brand_color=brand_color,
        image=image, audio=audio
    )
    ics_bytes = build_ics(project_name, gen_resp.schedule.model_dump())
    headers = {"Content-Disposition": f'attachment; filename="{project_name.replace(" ","_")}_schedule.ics"'}
    return StreamingResponse(BytesIO(ics_bytes), media_type="text/calendar", headers=headers)


@app.post("/export/csv")
async def export_csv(
    project_name: str = Form("Social Creative Coach"),
    brief: str = Form(""),
    channels: str = Form('["LinkedIn","Instagram","X"]'),
    brand_tone: str = Form("professionnel, clair, orienté valeur"),
    language: str = Form("fr"),
    target_audience: str = Form(""),
    call_to_action_hint: str = Form(""),
    forbid: str = Form("[]"),
    timezone: str = Form("Africa/Porto-Novo"),
    brand_name: str = Form(""),
    brand_color: str = Form(""),
    image: UploadFile | None = File(default=None),
    audio: UploadFile | None = File(default=None),
):
    """
    Génère les variantes puis renvoie un CSV (channel,title,body,hashtags,cta).
    """
    gen_resp: GenerateResponse = await generate(
        brief=brief, channels=channels, brand_tone=brand_tone, language=language,
        target_audience=target_audience, call_to_action_hint=call_to_action_hint,
        forbid=forbid, timezone=timezone, brand_name=brand_name, brand_color=brand_color,
        image=image, audio=audio
    )
    csv_bytes = build_csv(gen_resp.variants.model_dump())
    headers = {"Content-Disposition": f'attachment; filename="{project_name.replace(" ","_")}_posts.csv"'}
    return StreamingResponse(BytesIO(csv_bytes), media_type="text/csv", headers=headers)


@app.post("/export/md")
async def export_markdown(
    project_name: str = Form("Social Creative Coach"),
    brief: str = Form(""),
    channels: str = Form('["LinkedIn","Instagram","X"]'),
    brand_tone: str = Form("professionnel, clair, orienté valeur"),
    language: str = Form("fr"),
    target_audience: str = Form(""),
    call_to_action_hint: str = Form(""),
    forbid: str = Form("[]"),
    timezone: str = Form("Africa/Porto-Novo"),
    brand_name: str = Form(""),
    brand_color: str = Form(""),
    image: UploadFile | None = File(default=None),
    audio: UploadFile | None = File(default=None),
):
    """Génère puis renvoie un Markdown (posts + planning + image_prompt)."""
    gen_resp: GenerateResponse = await generate(
        brief=brief, channels=channels, brand_tone=brand_tone, language=language,
        target_audience=target_audience, call_to_action_hint=call_to_action_hint,
        forbid=forbid, timezone=timezone, brand_name=brand_name, brand_color=brand_color,
        image=image, audio=audio
    )
    md_bytes = build_markdown_export(project_name, gen_resp)
    headers = {"Content-Disposition": f'attachment; filename="{project_name.replace(" ","_")}.md"'}
    return StreamingResponse(BytesIO(md_bytes), media_type="text/markdown", headers=headers)


@app.post("/abtest")
async def abtest(
    channel: str = Form(..., description="Ex: LinkedIn|Instagram|X|Facebook|TikTok"),
    brief: str = Form("", description="Brief global"),
    brand_tone: str = Form("professionnel, clair, orienté valeur"),
    language: str = Form("fr"),
    hypothesis: str = Form("", description="Ex: plus d'urgence / plus d'emoji / ton plus fun"),
    image: UploadFile | None = File(default=None),
):
    """
    Génère A et B pour un canal puis renvoie une scorecard (qui gagne et pourquoi).
    Image facultative comme contexte.
    """
    # image en contexte si fournie
    parts = []
    image_used = False
    if image is not None:
        img_bytes = await image.read()
        if len(img_bytes) > IMAGE_MAX_BYTES:
            raise HTTPException(status_code=413, detail="Image trop lourde (> 20 Mo).")
        pil = load_image_from_bytes(img_bytes)
        if pil:
            image_used = True
            parts.append(pil)

    def build_abtest_variants_prompt(brief: str, channel: str, language: str, brand_tone: str, hypothesis: str) -> str:
        return f"""
Langue: {language}
Tu es un directeur créatif social media.
Brief: {brief}
Canal: {channel}
Ton de marque: {brand_tone}
Hypothèse B à tester: {hypothesis or "aucune (variante B = angle différent que A)"}

Génère DEUX variantes pour ce canal: A et B.
Réponds **strictement** en JSON valide, sans texte autour:
{{
  "A": {{"title":"...","body":"...","hashtags":["..."],"cta":"..."}},
  "B": {{"title":"...","body":"...","hashtags":["..."],"cta":"..."}}
}}

Rappels:
- Corps adapté au canal (longueur raisonnable).
- Hashtags nichés (pas génériques).
- Pas de champ null.
""".strip()

    def build_abtest_score_prompt(language: str) -> str:
        return f"""
Langue: {language}
Évalue objectivement deux variantes A et B d'un post pour un réseau social.
Renvoie un JSON **strict**:
{{
  "criteria": [
    {{"name":"Clarté du message","A":0-10,"B":0-10,"comment":"..."}},
    {{"name":"Pertinence pour l'audience","A":0-10,"B":0-10,"comment":"..."}},
    {{"name":"Force du hook/CTA","A":0-10,"B":0-10,"comment":"..."}},
    {{"name":"Adéquation au canal","A":0-10,"B":0-10,"comment":"..."}}
  ],
  "overall_winner":"A|B",
  "why":"raison en 1-2 phrases",
  "cta_reco":"CTA recommandé en 1 phrase"
}}
Barème: 0=très faible, 10=excellent. Sois concis.
""".strip()

    # 1) génère A/B
    try:
        prompt_ab = build_abtest_variants_prompt(
            brief=brief, channel=channel, language=language,
            brand_tone=brand_tone, hypothesis=hypothesis
        )
        resp_ab = model.generate_content(parts + [prompt_ab]) if parts else model.generate_content(prompt_ab)
        ab_raw = parse_json_safely(resp_ab.text or "")
        A = ab_raw.get("A", {}) if isinstance(ab_raw, dict) else {}
        B = ab_raw.get("B", {}) if isinstance(ab_raw, dict) else {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini abtest generation error: {e}")

    # 2) scorecard
    try:
        score_prompt = build_abtest_score_prompt(language=language)
        # On fournit A/B en amont du prompt de scoring
        score_input = json.dumps({"A": A, "B": B}, ensure_ascii=False)
        resp_score = model.generate_content([score_input, score_prompt])
        score_raw = parse_json_safely(resp_score.text or "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini abtest score error: {e}")

    return {"channel": channel, "image_used": image_used, "A": A, "B": B, "scorecard": score_raw}


@app.post("/image/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    language: str = Form("fr"),
):
    """Analyse d'image seule (caption/objets/couleurs/style/produit/humeur)."""
    img_bytes = await image.read()
    if len(img_bytes) > IMAGE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image trop lourde (> 20 Mo).")
    pil = load_image_from_bytes(img_bytes)
    if not pil:
        raise HTTPException(status_code=400, detail="Image illisible.")

    try:
        analysis_prompt = build_image_analysis_prompt(language)
        resp_analysis = model.generate_content([pil, analysis_prompt])
        analysis_raw = parse_json_safely(resp_analysis.text or "")
        image_summary = analysis_raw.get("image_summary", {}) if isinstance(analysis_raw, dict) else {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini image analysis error: {e}")

    return {"image_summary": image_summary}


# ----------------------
# Génération d'images réelles (ou placeholders) + ZIP
# ----------------------
def _placeholder_png(text: str, brand_color: Optional[str]) -> bytes:
    """Crée une image 1080x1080 simple avec texte centré."""
    w, h = 1080, 1080
    bg = "#0b1020"
    fg = "#e5e7eb"
    try:
        if brand_color and re.match(r"^#?[0-9a-fA-F]{6}$", brand_color):
            bg = brand_color if brand_color.startswith("#") else f"#{brand_color}"
    except Exception:
        pass
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default() if PLACEHOLDER_FONT is None else ImageFont.truetype(PLACEHOLDER_FONT, 36)
    # wrap basique
    lines = []
    words = text.split()
    line = ""
    for wd in words:
        trial = (line + " " + wd).strip()
        if len(trial) > 40:
            lines.append(line)
            line = wd
        else:
            line = trial
    if line:
        lines.append(line)
    msg = "\n".join(lines[:12]) or "Creative\nPlaceholder"
    tw, th = draw.multiline_textbbox((0, 0), msg, font=font, align="center")[2:]
    draw.multiline_text(((w - tw) / 2, (h - th) / 2), msg, fill=fg, font=font, align="center", spacing=6)
    out = BytesIO(); img.save(out, format="PNG"); return out.getvalue()


@app.post("/images/zip")
async def images_zip(
    project_name: str = Form("Social Creative Coach"),
    images_per_variant: int = Form(1),
    use_model: str = Form("auto"),  # "auto" | "gemini" | "placeholder"
    brand_name: str = Form(""),
    brand_color: str = Form(""),
    style_ref: UploadFile | None = File(default=None),

    # Reprend les mêmes inputs que /generate
    brief: str = Form(""),
    channels: str = Form('["LinkedIn","Instagram","X"]'),
    brand_tone: str = Form("professionnel, clair, orienté valeur"),
    language: str = Form("fr"),
    target_audience: str = Form(""),
    call_to_action_hint: str = Form(""),
    forbid: str = Form("[]"),
    timezone: str = Form("Africa/Porto-Novo"),
    image: UploadFile | None = File(default=None),
    audio: UploadFile | None = File(default=None),
):
    """
    Construit un ZIP d'images en s'appuyant sur /generate (variants + image_prompt).
    - Si GEMINI_IMAGE_MODEL dispo et use_model != "placeholder", tente une vraie génération.
    - Sinon, fallback placeholders.
    Structure du zip:
      README.md
      images/<channel>/<variant_index>_<k>.png
    """
    # 0) Récupérer variants + image_prompt avec /generate
    gen_resp: GenerateResponse = await generate(
        brief=brief, channels=channels, brand_tone=brand_tone, language=language,
        target_audience=target_audience, call_to_action_hint=call_to_action_hint,
        forbid=forbid, timezone=timezone, brand_name=brand_name, brand_color=brand_color,
        image=image, audio=audio
    )
    variants = gen_resp.variants.model_dump().get("variants", [])
    img_prompt = gen_resp.image_prompt.image_prompt

    # 1) Style reference (optionnel)
    style_ref_bytes = None
    if style_ref is not None:
        sb = await style_ref.read()
        if len(sb) > IMAGE_MAX_BYTES:
            raise HTTPException(status_code=413, detail="style_ref trop lourde (> 20 Mo).")
        style_ref_bytes = sb  # non utilisé dans placeholder; à brancher dans vrai appel modèle

    # 2) Nombre d’images par variante (1..4)
    n = max(1, min(int(images_per_variant or 1), 4))

    # 3) Préparer le zip en mémoire
    mem = BytesIO()
    with ZipFile(mem, "w", ZIP_DEFLATED) as z:
        # README
        readme = [
            f"# {project_name} — Kit Images",
            "",
            f"- Date: {datetime.utcnow().isoformat()}Z",
            f"- Brand name: {brand_name or 'N/A'}",
            f"- Brand color: {brand_color or 'N/A'}",
            f"- Image model: {GEMINI_IMAGE_MODEL or 'placeholder'}",
            "",
            "## Prompt d’image de référence",
            "",
            "```",
            img_prompt,
            "```",
            "",
            "## Variantes (texte)",
            "",
        ]
        for v in variants:
            ch = v.get("channel", "")
            title = v.get("title", "")
            body = v.get("body", "")
            tags = " ".join("#"+t for t in (v.get("hashtags", []) or []))
            readme += [f"### {ch}", f"**{title}**", "", body, "", tags, ""]
        z.writestr("README.md", "\n".join(readme))

        # 4) Génération réelle si possible
        real_model_available = GEMINI_IMAGE_MODEL and (use_model in ("auto", "gemini"))
        image_model = None
        if real_model_available:
            try:
                image_model = genai.GenerativeModel(GEMINI_IMAGE_MODEL)  # peut lever si modèle indispo
            except Exception:
                image_model = None

        for idx, v in enumerate(variants):
            ch = v.get("channel", "Unknown")
            ch_dir = f"images/{ch}"
            for k in range(1, n+1):
                filename = f"{ch_dir}/{idx+1}_{k}.png"
                # Par défaut: placeholder
                png_bytes = _placeholder_png(
                    text=f"{(brand_name or '').strip()}\n{ch} v{idx+1}-{k}",
                    brand_color=brand_color or None
                )

                # Tentative génération réelle
                if image_model is not None:
                    try:
                        # ⚠️ NOTE IMPORTANTE:
                        # La génération d'image via le SDK peut différer selon la version.
                        # Ci-dessous on passe par un appel textuel; si votre SDK expose une API dédiée,
                        # remplacez par l'appel officiel (ex: image_model.generate_images(...) si disponible).
                        full_prompt = (
                            f"Crée une image 1080x1080 pour {brand_name or 'la marque'}, "
                            f"couleur dominante {brand_color or 'neutre'}. "
                            f"Contenu: {img_prompt}. "
                            f"Variante {k} pour le post {idx+1} sur {ch}."
                        )
                        # Beaucoup de SDK renvoient des textes; si le vôtre renvoie des images, adaptez:
                        resp = image_model.generate_content(full_prompt)
                        # Si pas d'image binaire retournée, on garde le placeholder
                        # (Tu peux parser resp pour extraire des bytes si nécessaire)
                    except Exception:
                        pass  # fallback placeholder

                z.writestr(filename, png_bytes)

    mem.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{project_name.replace(" ","_")}_images.zip"'}
    return StreamingResponse(mem, media_type="application/zip", headers=headers)


# ----------------------
# Stub vidéo (bonus)
# ----------------------
@app.get("/video/teaser")
def video_teaser():
    """
    Sert un teaser MP4 d'exemple depuis public/samples/teaser.mp4 si présent.
    """
    base = Path(__file__).resolve().parent.parent
    p = base / "public" / "samples" / "teaser.mp4"
    if p.is_file():
        return FileResponse(str(p), media_type="video/mp4", filename="teaser.mp4")
    raise HTTPException(status_code=501, detail="Aucune vidéo d'exemple disponible.")


# ----------------------
# Error handler
# ----------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


# ----------------------
# UI statique (public/index.html)
# ----------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PUBLIC_DIR = BASE_DIR / "public"
if PUBLIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="static")
