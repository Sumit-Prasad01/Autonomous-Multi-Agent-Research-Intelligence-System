# Quick check — what sections does TurboQuant have?
from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333")
results, _ = client.scroll(
    collection_name="research_papers",
    limit=100,
    with_payload=True,
    with_vectors=False,
)
from collections import Counter
sections = Counter(
    r.payload.get("metadata", {}).get("section", "unknown") 
    for r in results
)
print(dict(sections))