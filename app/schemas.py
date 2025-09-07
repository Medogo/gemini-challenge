# app/schemas.py
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Channel(str, Enum):
    LinkedIn = "LinkedIn"
    Instagram = "Instagram"
    X = "X"
    Facebook = "Facebook"
    TikTok = "TikTok"

class GenerateRequest(BaseModel):
    brief: str = Field(..., description="Description de la campagne / idée")
    channels: List[Channel] = Field(default=[Channel.LinkedIn, Channel.Instagram, Channel.X])
    brand_tone: str = "professionnel, clair, orienté valeur"
    language: str = "fr"
    target_audience: Optional[str] = None
    call_to_action_hint: Optional[str] = None
    forbid: List[str] = []
    timezone: str = "Africa/Porto-Novo"

    # 🔥 Nouveaux champs marque (pour prompts texte & génération d’images)
    brand_name: Optional[str] = None
    brand_color: Optional[str] = None  # Attendu sous forme hex ex. "#22c55e"

class VariantItem(BaseModel):
    channel: str
    title: str
    body: str
    hashtags: List[str]
    cta: str

class Variants(BaseModel):
    variants: List[VariantItem]

class ScheduleSlot(BaseModel):
    iso: str
    why: str

class ChannelSchedule(BaseModel):
    channel: str
    slots: List[ScheduleSlot]

class ScheduleResponse(BaseModel):
    schedule: List[ChannelSchedule]

class ImagePrompt(BaseModel):
    image_prompt: str

class GenerateResponse(BaseModel):
    variants: Variants
    schedule: ScheduleResponse
    image_prompt: ImagePrompt
    input_image_used: bool = False
    input_audio_transcript: Optional[str] = None
    # 🔥 Ajouté (retourné par /generate quand image seule sert de brief)
    image_analysis: Optional[Dict[str, Any]] = None
