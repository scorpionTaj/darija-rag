import argparse
import json
import os
import sys
import time
from pathlib import Path

from darija_rag.evaluate import (
    answerable,
    group,
    load_questions,
    metrics,
    ranks,
    stratified_sample,
)
from darija_rag.ingest import ingest, load_chunks
from darija_rag.llm import LLMClient, LLMError

DEFAULT_CHUNKS = Path("data/processed/chunks.jsonl")
SPEC_HELP = "retriever spec: bm25|dense|dense-m3|hybrid|hybrid-m3, plus +rewrite and/or +rerank"


def _factory(chunks_path: Path):
    from darija_rag.factory import Factory

    if not chunks_path.exists():
        sys.exit(f"{chunks_path} not found; run `darija-rag ingest --ocr '*_ar.pdf'` first")
    return Factory(load_chunks(chunks_path), cache_dir=chunks_path.parent)


def _answerer(factory, spec: str, k: int):
    from darija_rag.answer import Answerer

    rewriter = factory.rewriter() if factory.uses_rewrite(spec) else None
    return Answerer(factory.build(spec), factory.llm, k=k, rewriter=rewriter)


def _print_answer(answer) -> None:
    if answer.retrieval_query != answer.question:
        print(f"(searched as: {answer.retrieval_query})")
    print(answer.text)
    if answer.cited_chunks:
        print("\nSources:")
        for i in answer.cited:
            chunk = answer.sources[i]
            print(f"  [{i + 1}] {chunk.doc} p.{chunk.page}: {chunk.text[:160]}")
    print(f"\n({answer.timings['total_s']:.1f}s, {answer.variety})")


def cmd_eval(args) -> None:
    questions = load_questions(args.questions)
    factory = _factory(args.chunks)
    groups = group(answerable(questions))
    print("| Retriever | Questions | n | Recall@1 | Recall@5 | Recall@10 | MRR@10 |")
    print("|---|---|---|---|---|---|---|")
    misses = {}
    for spec in args.retriever or ["bm25"]:
        question_ranks = ranks(factory.build(spec).retrieve, groups["all"])
        for name, subset in groups.items():
            r = metrics({q["id"]: question_ranks[q["id"]] for q in subset})
            print(
                f"| {spec} | {name} | {r['n']} | {r['recall@1']:.2f} | {r['recall@5']:.2f}"
                f" | {r['recall@10']:.2f} | {r['mrr@10']:.2f} |",
                flush=True,
            )
            if name == "all":
                misses[spec] = r["misses"]
    if args.misses:
        args.misses.write_text(json.dumps(misses, ensure_ascii=False, indent=2), "utf-8")


def cmd_eval_answers(args) -> None:
    from darija_rag.judge import answer_table, judge_answer, score

    questions = load_questions(args.questions)
    if args.sample:
        questions = stratified_sample(questions, args.sample, seed=args.seed)
    elif args.limit:
        questions = questions[: args.limit]

    # --resume keeps rows already in --out (a killed run loses nothing) and skips their ids.
    done = {}
    if args.resume and args.out.exists():
        done = {r["id"]: r for r in map(json.loads, args.out.read_text("utf-8").splitlines()) if r}
    todo = [q for q in questions if q["id"] not in done]
    print(f"{len(questions) - len(todo)} already done, {len(todo)} to run", file=sys.stderr)

    factory = _factory(args.chunks)
    judge = None
    if todo:
        answerer = _answerer(factory, args.retriever, args.k)
        if not args.no_judge:
            judge = LLMClient.from_env()
            judge.model = (
                os.environ.get("DARIJA_RAG_JUDGE_MODEL", "") or factory.llm.resolve_model()
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a" if args.resume else "w", encoding="utf-8") as out:
        for number, q in enumerate(todo, start=1):
            start = time.perf_counter()
            answer = answerer.ask(q["question"])
            verdict = None
            if judge and q["relevant"] and not answer.abstained:
                verdict = judge_answer(judge, q, answer)
            row = score(q, answer, verdict)
            done[q["id"]] = row
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"{number}/{len(todo)} {q['id']} correct={row['correct']} "
                f"({time.perf_counter() - start:.0f}s incl. judge)",
                file=sys.stderr,
            )

    rows = [done[q["id"]] for q in questions if q["id"] in done]
    print(
        f"Retriever: {args.retriever}, generator: {factory.llm.base_url} "
        f"{factory.llm.model or '(not called)'}, judge: {judge.model if judge else 'none'}\n"
    )
    print(answer_table(rows))
    print(f"\nPer-question results: {args.out}")


def cmd_report(args) -> None:
    from darija_rag.judge import answer_table

    rows = [json.loads(line) for line in args.results.read_text("utf-8").splitlines() if line]
    print(answer_table(rows))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="darija-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="extract and chunk PDFs/text files")
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=DEFAULT_CHUNKS)
    p.add_argument("--size", type=int, default=200, help="words per chunk")
    p.add_argument("--overlap", type=int, default=50, help="words shared by neighbours")
    p.add_argument(
        "--ocr",
        action="append",
        default=[],
        metavar="GLOB",
        help="OCR PDFs whose file name matches (repeatable), e.g. --ocr '*_ar.pdf'",
    )
    p.add_argument("--ocr-lang", default="ara", help="tesseract language code")

    p = sub.add_parser("search", help="search ingested chunks")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    p.add_argument("--retriever", default="bm25", help=SPEC_HELP)

    p = sub.add_parser("ask", help="answer a question with citations (needs an LLM endpoint)")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=5, help="sources given to the LLM")
    p.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    p.add_argument("--retriever", default="dense-m3+rewrite+rerank", help=SPEC_HELP)

    p = sub.add_parser("eval", help="retrieval recall@k and MRR, by language and question type")
    p.add_argument("questions", type=Path)
    p.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    p.add_argument("--retriever", action="append", help=f"repeatable; {SPEC_HELP}")
    p.add_argument("--misses", type=Path, help="write missed question ids as JSON here")

    p = sub.add_parser("eval-answers", help="end-to-end answer accuracy, faithfulness, latency")
    p.add_argument("questions", type=Path)
    p.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    p.add_argument("--retriever", default="dense-m3+rewrite+rerank", help=SPEC_HELP)
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    p.add_argument(
        "--sample", type=int, help="N questions spread evenly over language and type (fast run)"
    )
    p.add_argument("--seed", type=int, default=0, help="seed for --sample")
    p.add_argument("--resume", action="store_true", help="keep rows already in --out, run the rest")
    p.add_argument(
        "--no-judge", action="store_true", help="skip LLM judging (abstention/citations only)"
    )
    p.add_argument("--out", type=Path, default=Path("data/processed/answers.jsonl"))

    p = sub.add_parser("report", help="answer-eval table from a results file, no LLM needed")
    p.add_argument("results", type=Path, nargs="?", default=Path("data/processed/answers.jsonl"))

    p = sub.add_parser("serve", help="run the HTTP API and chat UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    try:
        if args.command == "ingest":
            count = ingest(args.raw, args.out, args.size, args.overlap, args.ocr, args.ocr_lang)
            print(f"wrote {count} chunks to {args.out}")
        elif args.command == "search":
            retriever = _factory(args.chunks).build(args.retriever)
            for rank, (chunk, score) in enumerate(retriever.retrieve(args.query, args.k), start=1):
                print(f"{rank}. [{score:.3f}] {chunk.doc} p.{chunk.page}\n   {chunk.text[:240]}\n")
        elif args.command == "ask":
            _print_answer(
                _answerer(_factory(args.chunks), args.retriever, args.k).ask(args.question)
            )
        elif args.command == "eval":
            cmd_eval(args)
        elif args.command == "eval-answers":
            cmd_eval_answers(args)
        elif args.command == "report":
            cmd_report(args)
        elif args.command == "serve":
            import uvicorn

            uvicorn.run("darija_rag.api:app", host=args.host, port=args.port)
    except (LLMError, ValueError) as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
