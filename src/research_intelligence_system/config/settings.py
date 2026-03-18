from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):

    GROQ_API_KEY: str =  os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY")
    HF_TOKEN : str = os.getenv("HF_TOKEN")
    HUGGINFACEHUB_API_TOKEN : str = os.getenv("HUGGINFACEHUB_API_TOKEN")

    DEFAULT_LLM: str = "llama-3.1-8b-instant"

    RETRIEVAL_K: int = 5
    RETRIEVAL_LAMBDA: float = 0.7

    TAVILY_MAX_RESULTS: int = 3

    MAX_ANSWER_SENTENCES: int = 3

    class Config:
        env_file = ".env"


settings = Settings()