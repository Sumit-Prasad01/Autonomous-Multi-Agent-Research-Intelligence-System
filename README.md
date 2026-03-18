# Autonomous-Multi-Agent-Research-Intelligence-System
```
Research Paper
      ↓
PDF Parser
      ↓
Chunking
      ↓
Classification Model
      ↓
Paper Category
      ↓
Summarization Model
      ↓
Structured Summary
      ↓
Vector DB
      ↓
RAG System
      ↓
Agents
```


```
User Query
   ↓
Router
   ↓
 ├── RAG (Primary)
 │       ↓
 │   If confident → Return
 │
 └── Web Search (Fallback)
         ↓
     Combine + Answer
         ↓
Structured Output
```

```
Final Design (Production Grade)
FastAPI
   ↓
chat_service
   ↓
run_qa (router)
   ↓
 ├── RAG (FAISS + compression)
 ├── Tavily (optional)
 └── LLM (final synthesis)
   ↓
Structured JSON Output
```