import os
from dotenv import load_dotenv

load_dotenv()

class Settings:

    GROQ_API_KEY = os.getenv("GROQ_PAI_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")
    HUGGINFACEHUB_API_TOKEN = os.getenv("HUGGINFACEHUB_API_TOKEN")


settings = Settings()