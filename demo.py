"""End-to-end CLI demo: ingest the sample knowledge base and answer questions.

    python demo.py            # Ollama if up, else offline
    RAG_OFFLINE=1 python demo.py
"""

from pathlib import Path

from charterqa.core.pipeline import CharterQaPipeline

QUESTIONS = [
    "How long do I have to request a refund?",
    "What is the SLA for priority incidents?",
    "How long is personal data retained?",
]


def main() -> None:
    doc = Path("data/sample_knowledge_base.md").read_text(encoding="utf-8")
    pipe = CharterQaPipeline()
    n = pipe.ingest(doc, source="sample_knowledge_base.md")
    print(f"Ingested {n} chunks (backend: {type(pipe.llm).__name__})\n")

    for q in QUESTIONS:
        result = pipe.ask(q)
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        if result["sources"]:
            top = result["sources"][0]
            print(f"   [top: {top['source']} fusion={top['fusion_score']} ranks={top['retriever_ranks']}]")
        print()


if __name__ == "__main__":
    main()
