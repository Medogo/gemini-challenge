import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.5-flash"
    TIMEZONE: str = "Africa/Porto-Novo"
    MAX_CONTENT_LENGTH_MB: int = 20  # simple garde-fou

    class Config:
        env_file = ".env"

settings = Settings()
