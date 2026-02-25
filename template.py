import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "research_intelligence_system"

list_of_files = [

    # GitHub
    ".github/workflows/.gitkeep",

    # Source root
    f"src/{project_name}/__init__.py",

    # ---------------- CORE MODULES ----------------
    f"src/{project_name}/agents/__init__.py",
    f"src/{project_name}/agents/parsing_agent.py",
    f"src/{project_name}/agents/extraction_agent.py",
    f"src/{project_name}/agents/summarizer_agent.py",
    f"src/{project_name}/agents/critic_agent.py",
    f"src/{project_name}/agents/comparison_agent.py",
    f"src/{project_name}/agents/literature_agent.py",

    # ---------------- RAG ----------------
    f"src/{project_name}/rag/__init__.py",
    f"src/{project_name}/rag/embedding.py",
    f"src/{project_name}/rag/vector_store.py",
    f"src/{project_name}/rag/retriever.py",

    # ---------------- KNOWLEDGE GRAPH ----------------
    f"src/{project_name}/knowledge_graph/__init__.py",
    f"src/{project_name}/knowledge_graph/triple_extractor.py",
    f"src/{project_name}/knowledge_graph/graph_builder.py",

    # ---------------- MODELS ----------------
    f"src/{project_name}/models/__init__.py",
    f"src/{project_name}/models/summarization_model.py",
    f"src/{project_name}/models/ner_model.py",
    f"src/{project_name}/models/classification_model.py",

    # ---------------- PIPELINES ----------------
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/paper_processing_pipeline.py",
    f"src/{project_name}/pipeline/query_pipeline.py",

    # ---------------- UTILITIES ----------------
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/pdf_parser.py",
    f"src/{project_name}/utils/chunking.py",
    f"src/{project_name}/utils/common.py",

    # ---------------- CONFIG ----------------
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",

    # ---------------- ENTITY ----------------
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/paper_entity.py",

    # ---------------- CONSTANTS ----------------
    f"src/{project_name}/constants/__init__.py",

    # ---------------- EVALUATION ----------------
    f"src/{project_name}/evaluation/__init__.py",
    f"src/{project_name}/evaluation/metrics.py",

    # ---------------- CONFIG FILES ----------------
    "config/config.yaml",
    "params.yaml",

    # ---------------- APP ENTRY ----------------
    "app.py",
    "main.py",

    # ---------------- DEVOPS ----------------
    "Dockerfile",
    "requirements.txt",
    "setup.py",

    # ---------------- RESEARCH ----------------
    "research/experiments.ipynb",
    "research/ablation_study.ipynb",

    # ---------------- DATA ----------------
    "artifacts/.gitkeep"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w'):
            pass
        logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} already exists")