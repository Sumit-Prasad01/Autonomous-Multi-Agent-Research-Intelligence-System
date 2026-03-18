from fastapi import FastAPI
import uvicorn

from api.chat import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Research Intelligence API",
        description="RAG-based QA system with optional web search using Groq and FAISS",
        version="1.0.0"
    )

    app.include_router(router, prefix="/api")

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",   
        port=8000,
        reload=False
    )