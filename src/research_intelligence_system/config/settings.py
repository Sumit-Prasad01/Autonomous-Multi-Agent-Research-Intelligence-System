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

    DEFAULT_LLM: str = "llama-3.1-8b-instant"

    ALLOWED_MODELS : List[str] = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile"
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b"
    ]

    RETRIEVAL_K: int = 5
    RETRIEVAL_LAMBDA: float = 0.7

    TAVILY_MAX_RESULTS: int = 3

    MAX_ANSWER_SENTENCES: int = 3

    class Config:
        env_file = ".env"


settings = Settings()