from pydantic_settings import BaseSettings
import os
from typing import List
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):

    GROQ_API_KEY: str =  os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY")
    HF_TOKEN : str = os.getenv("HF_TOKEN")
    HUGGINFACEHUB_API_TOKEN : str = os.getenv("HUGGINFACEHUB_API_TOKEN")
    FRONTEND_ORIGIN_URL : str = os.getenv("FRONTEND_ORIGIN_URL")
    BACKEND_ORIGIN_URL : str = os.getenv("BACKEND_ORIGIN_URL")
    DATABASE_URL : str = os.getenv("DATABASE_URL")
    QDRANT_URL : str = os.getenv("QDRANT_URL")
    SECRET_KEY : str = os.getenv("SECRET_KEY")
    REDIS_URL : str = os.getenv("REDIS_URL")
    COOKIE_SECRET : str = os.getenv("COOKIE_SECRET")
    NEO4J_URI : str = os.getenv("NEO4J_URI")
    NEO4J_USER : str = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD : str = os.getenv("NEO4J_PASSWORD")

    class Config:
        env_file = ".env"


settings = Settings()