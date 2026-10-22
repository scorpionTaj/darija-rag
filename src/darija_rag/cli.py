import argparse
import json
from pathlib import Path

from darija_rag.evaluate import evaluate, load_questions
from darija_rag.ingest import ingest, load_chunks
from darija_rag.retrieval import BM25Retriever

DEFAULT_CHUNKS = Path("data/processed/chunks.jsonl")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="darija-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="extract and chunk PDFs/text files")
    p_ingest.add_argument("--raw", type=Path, default=Path("data/raw"))
    p_ingest.add_argument("--out", type=Path, default=DEFAULT_CHUNKS)
    p_ingest.add_argument("--size", type=int, default=200, help="words per chunk")
    p_ingest.add_argument("--overlap", type=int, default=50, help="words shared by neighbours")

    p_search = sub.add_parser("search", help="BM25 search over ingested chunks")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)
    p_search.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)

    p_eval = sub.add_parser("eval", help="recall@k and MRR on a labeled question set")
    p_eval.add_argument("questions", type=Path)
    p_eval.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)

    args = parser.parse_args(argv)

    if args.command == "ingest":
        count = ingest(args.raw, args.out, args.size, args.overlap)
        print(f"wrote {count} chunks to {args.out}")
    elif args.command == "search":
        retriever = BM25Retriever(load_chunks(args.chunks))
        for rank, (chunk, score) in enumerate(retriever.retrieve(args.query, args.k), start=1):
            print(f"{rank}. [{score:.2f}] {chunk.doc} p.{chunk.page}\n   {chunk.text[:240]}\n")
    elif args.command == "eval":
        retriever = BM25Retriever(load_chunks(args.chunks))
        questions = load_questions(args.questions)
        report = {"all": evaluate(retriever.retrieve, questions)}
        for lang in sorted({q["lang"] for q in questions if "lang" in q}):
            subset = [q for q in questions if q.get("lang") == lang]
            report[lang] = evaluate(retriever.retrieve, subset)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
