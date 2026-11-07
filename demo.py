"""End-to-end CLI demo: ingest the sample charter party and answer questions.

    python demo.py            # Ollama if up, else offline
    RAG_OFFLINE=1 python demo.py
"""

from pathlib import Path

from charterqa.core.pipeline import CharterQaPipeline

QUESTIONS = [
    "What is the demurrage rate if unloading is late?",
    "How much laytime is allowed for loading and discharging?",
    "When is freight payable and at what rate?",
]


def main() -> None:
    doc = Path("data/voyage_charter.md").read_text(encoding="utf-8")
    pipe = CharterQaPipeline()
    n = pipe.ingest(doc, source="voyage_charter.md")
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
