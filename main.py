from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    "data/python_intro.txt",
    "data/vector_store_notes.md",
    "data/rag_system_design.md",
    "data/customer_support_playbook.txt",
    "data/chunking_experiment_report.md",
    "data/vi_retrieval_notes.md",
]


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def get_llm_fn():
    """Returns Gemini LLM generator if configured, else fallback to demo_llm."""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            model_name = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash")

            def gemini_llm(prompt: str) -> str:
                try:
                    res = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    return res.text.strip()
                except Exception:
                    return demo_llm(prompt)

            return gemini_llm
        except Exception:
            pass
    return demo_llm


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "gemini":
        try:
            embedder = GeminiEmbedder(model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    llm = get_llm_fn()
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3))
    return 0


def run_university_demo(question: str | None = None) -> int:
    """Run RAG demonstration using cleaned FPTU university documents and HeadingAwareContextChunker."""
    query = question or "Học bổng FPTU có những mức nào và duy trì ra sao?"
    print("=== FPTU University Regulations RAG Test ===")
    from src.custom_chunker import HeadingAwareContextChunker

    chunker = HeadingAwareContextChunker(max_chunk_size=1000, overlap=100)
    uni_dir = Path("data/university")
    if not uni_dir.exists():
        print(f"Directory {uni_dir} does not exist.")
        return 1

    chunks: list[Document] = []
    for md_path in sorted(uni_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        raw_chunks = chunker.chunk(text)
        for i, chunk_text in enumerate(raw_chunks):
            chunks.append(
                Document(
                    id=f"{md_path.stem}_{i}",
                    content=chunk_text,
                    metadata={"source": md_path.name, "doc_id": md_path.stem},
                )
            )

    print(f"Loaded {len(chunks)} chunks from {len(list(uni_dir.glob('*.md')))} university documents.")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "gemini":
        try:
            embedder = GeminiEmbedder(model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"Embedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="fptu_regulations_store", embedding_fn=embedder)
    store.add_documents(chunks)
    print(f"Stored {store.get_collection_size()} chunks in EmbeddingStore")

    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    results = store.search(query, top_k=3)
    for index, r in enumerate(results, start=1):
        print(f"{index}. score={r['score']:.3f} source={r['metadata'].get('source')}")
        print(f"   content preview: {r['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    llm = get_llm_fn()
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3))
    return 0


def main() -> int:
    args = sys.argv[1:]
    if "--university" in args or "-u" in args:
        filtered = [a for a in args if a not in ("--university", "-u")]
        question = " ".join(filtered).strip() if filtered else None
        return run_university_demo(question=question)
    question = " ".join(args).strip() if args else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())
