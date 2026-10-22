import argparse
import json
from pathlib import Path

from darija_rag.evaluate import evaluate, load_questions
from darija_rag.ingest import Chunk, ingest, load_chunks
from darija_rag.retrieval import BM25Retriever, HybridRetriever, Retriever

DEFAULT_CHUNKS = Path("data/processed/chunks.jsonl")
RETRIEVERS = ("bm25", "dense", "hybrid")


def build_retrievers(names: list[str], chunks: list[Chunk]) -> dict[str, Retriever]:
    """Build each requested retriever once; hybrid reuses the BM25 and dense instances."""
    built: dict[str, Retriever] = {}

    def get(name: str) -> Retriever:
        if name not in built:
            if name == "bm25":
                built[name] = BM25Retriever(chunks)
            elif name == "dense":
                from darija_rag.dense import DenseRetriever

                built[name] = DenseRetriever(chunks)
            else:
                built[name] = HybridRetriever([get("bm25"), get("dense")])
        return built[name]

    return {name: get(name) for name in names}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="darija-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="extract and chunk PDFs/text files")
    p_ingest.add_argument("--raw", type=Path, default=Path("data/raw"))
    p_ingest.add_argument("--out", type=Path, default=DEFAULT_CHUNKS)
    p_ingest.add_argument("--size", type=int, default=200, help="words per chunk")
    p_ingest.add_argument("--overlap", type=int, default=50, help="words shared by neighbours")
    p_ingest.add_argument(
        "--ocr",
        action="append",
        default=[],
        metavar="GLOB",
        help="OCR PDFs whose file name matches (repeatable), e.g. --ocr '*_ar.pdf'",
    )
    p_ingest.add_argument("--ocr-lang", default="ara", help="tesseract language code")

    p_search = sub.add_parser("search", help="search ingested chunks")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)
    p_search.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    p_search.add_argument("--retriever", choices=RETRIEVERS, default="bm25")

    p_eval = sub.add_parser("eval", help="recall@k and MRR on a labeled question set")
    p_eval.add_argument("questions", type=Path)
    p_eval.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    p_eval.add_argument(
        "--retriever", action="append", choices=RETRIEVERS, help="repeatable; default bm25"
    )
    p_eval.add_argument("--misses", type=Path, help="write missed questions as JSON here")

    args = parser.parse_args(argv)

    if args.command == "ingest":
        count = ingest(args.raw, args.out, args.size, args.overlap, args.ocr, args.ocr_lang)
        print(f"wrote {count} chunks to {args.out}")
    elif args.command == "search":
        retriever = build_retrievers([args.retriever], load_chunks(args.chunks))[args.retriever]
        for rank, (chunk, score) in enumerate(retriever.retrieve(args.query, args.k), start=1):
            print(f"{rank}. [{score:.2f}] {chunk.doc} p.{chunk.page}\n   {chunk.text[:240]}\n")
    elif args.command == "eval":
        questions = load_questions(args.questions)
        retrievers = build_retrievers(args.retriever or ["bm25"], load_chunks(args.chunks))
        groups = {"all": questions}
        for lang in sorted({q["lang"] for q in questions if "lang" in q}):
            groups[lang] = [q for q in questions if q.get("lang") == lang]
        print("| Retriever | Questions | n | Recall@1 | Recall@5 | Recall@10 | MRR@10 |")
        print("|---|---|---|---|---|---|---|")
        misses = {}
        for name, retriever in retrievers.items():
            for group, subset in groups.items():
                r = evaluate(retriever.retrieve, subset)
                print(
                    f"| {name} | {group} | {r['n']} | {r['recall@1']:.2f} | {r['recall@5']:.2f}"
                    f" | {r['recall@10']:.2f} | {r['mrr@10']:.2f} |"
                )
                if group == "all":
                    misses[name] = r["misses"]
        if args.misses:
            args.misses.write_text(json.dumps(misses, ensure_ascii=False, indent=2), "utf-8")


if __name__ == "__main__":
    main()
