import re
from typing import List
from langchain_core.documents import Document


# ---------- SECTION PATTERNS ----------
SECTION_PATTERNS = {
    "abstract": r"\babstract\b",
    "introduction": r"\bintroduction\b",
    "methodology": r"\b(methodology|methods|approach|framework)\b",
    "results": r"\b(results|experiments|evaluation|findings)\b",
    "conclusion": r"\b(conclusion|discussion|future work|limitations)\b"
}


# ---------- CLEAN HEADER ----------
def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


class ParsingAgent:
    """
    Production Parsing Agent:
    - Section-aware parsing
    - Robust detection
    - Noise-resistant
    """

    def __init__(self):
        self.section_patterns = {
            key: re.compile(pattern, re.IGNORECASE)
            for key, pattern in SECTION_PATTERNS.items()
        }

    def detect_section(self, text: str, current_section: str) -> str:
        """
        Detect section from text header
        """

        # check only first part (header zone)
        header = text[:300].lower()

        for section, pattern in self.section_patterns.items():
            if pattern.search(header):
                return section

        return current_section

    def parse_documents(self, documents: List[Document]) -> List[Document]:
        """
        Add section metadata to each document
        """

        parsed_docs = []
        current_section = "unknown"

        for doc in documents:
            raw_text = doc.page_content

            if not raw_text:
                continue

            text = normalize_text(raw_text)

            if len(text) < 40:
                continue

            # detect section
            detected_section = self.detect_section(text, current_section)
            current_section = detected_section

            # enrich metadata
            metadata = {
                **(doc.metadata or {}),
                "section": current_section
            }

            parsed_docs.append(
                Document(
                    page_content=text,
                    metadata=metadata
                )
            )

        return parsed_docs