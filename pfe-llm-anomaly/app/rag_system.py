import os, json, re, logging
from pathlib import Path
import numpy as np

log = logging.getLogger(__name__)

KNOWLEDGE_BASE_DIR = Path(os.getenv("KNOWLEDGE_BASE_DIR", "/teamspace/studios/this_studio/pfe-llm-anomaly/knowledge_base"))
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 3
EMBED_MODEL = "all-MiniLM-L6-v2"

def read_document(path):
    ext = path.suffix.lower()
    if ext in [".md", ".txt"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""

def chunk_document(text, source):
    if not text.strip(): return []
    sections = re.split(r'\n(?=#{1,3} )', text)
    chunks = []
    for section in sections:
        words = section.split()
        if not words: continue
        title = re.sub(r'^#+\s*', '', section.strip().split('\n')[0]).strip() or source
        start = 0
        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))
            chunks.append({"text": ' '.join(words[start:end]), "source": source, "title": title[:80]})
            if end == len(words): break
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

class FAISSKnowledgeBase:
    def __init__(self):
        self.chunks = []
        self.index = None
        self.model = None
        self.is_ready = False
        self.stats = {}

    def build(self, kb_dir=KNOWLEDGE_BASE_DIR):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("Missing packages: pip install faiss-cpu sentence-transformers")
            return self

        self.model = SentenceTransformer(EMBED_MODEL)
        self.chunks = []
        all_files = list(kb_dir.glob("*.md")) + list(kb_dir.glob("*.txt"))

        for fp in sorted(all_files):
            text = read_document(fp)
            if not text.strip(): continue
            chunks = chunk_document(text, fp.name)
            self.chunks.extend(chunks)
            self.stats[fp.name] = {"chunks": len(chunks)}

        if not self.chunks:
            return self

        embeddings = self.model.encode([c["text"] for c in self.chunks], batch_size=32, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.is_ready = True
        print(f"FAISS ready — {self.index.ntotal} vectors")
        return self

    def retrieve(self, query, k=TOP_K):
        if not self.is_ready: return []
        qv = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(qv, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0: continue
            c = self.chunks[idx].copy()
            c["score"] = round(float(score), 4)
            results.append(c)
        return results

    def format_context(self, chunks):
        if not chunks: return ""
        return "\n\n".join(f"[Rule {i} — {c['source']}, section: {c['title']}]\n{c['text']}" for i, c in enumerate(chunks, 1))

kb = FAISSKnowledgeBase()

def build_rag_prompt(record, issues, anomaly_type=None, topic="retail.orders"):
    """
    Build a (system_prompt, user_prompt) pair for anomaly explanation,
    augmented with retrieved rules/context from the knowledge base.
    """
    issues_text = "; ".join(str(i) for i in issues) if issues else "unspecified anomaly"
    query = f"{topic} {anomaly_type or ''} {issues_text}".strip()

    retrieved = kb.retrieve(query, k=TOP_K)
    context = kb.format_context(retrieved)

    system_prompt = (
        "You are a data quality expert analyzing anomalies in "
        f"{topic} records. "
        "Use the reference rules below (if any) to ground your explanation. "
        "Respond ONLY with a valid JSON object with the fields: "
        "\"explanation\" (string), \"root_cause\" (string), "
        "\"severity\" (one of: low, medium, high, critical), "
        "\"recommended_action\" (string). "
        "Do not include any text outside the JSON object."
    )

    if context:
        system_prompt += f"\n\nReference rules:\n{context}"

    user_prompt = (
        f"Anomaly type: {anomaly_type or 'unknown'}\n"
        f"Detected issues: {issues_text}\n"
        f"Record:\n{json.dumps(record, indent=2, default=str)}\n\n"
        "Explain this anomaly and provide the JSON response as instructed."
    )

    return system_prompt, user_prompt
