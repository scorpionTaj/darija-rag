"""Retrieval metrics against a labeled question set.

Each line of the question file is JSON: {"question": ..., "doc": "file.pdf", "page": 12}.
A retrieved chunk counts as relevant when it comes from the labeled doc and page.
"""

import json
from collections.abc import Callable
from pathlib import Path

from darija_rag.ingest import Chunk

Retrieve = Callable[[str, int], list[tuple[Chunk, float]]]


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    for number, q in enumerate(questions, start=1):
        missing = {"question", "doc", "page"} - q.keys()
        if missing:
            raise ValueError(f"{path}:{number} missing fields {sorted(missing)}")
    return questions


def evaluate(retrieve: Retrieve, questions: list[dict], ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    if not questions:
        raise ValueError("no questions to evaluate")
    depth = max(ks)
    hits = {k: 0 for k in ks}
    reciprocal_rank_sum = 0.0
    misses = []
    for q in questions:
        results = retrieve(q["question"], depth)
        rank = next(
            (
                position
                for position, (chunk, _) in enumerate(results, start=1)
                if chunk.doc == q["doc"] and chunk.page == q["page"]
            ),
            None,
        )
        if rank is None:
            misses.append(q["question"])
            continue
        reciprocal_rank_sum += 1 / rank
        for k in ks:
            hits[k] += rank <= k
    n = len(questions)
    report = {f"recall@{k}": hits[k] / n for k in ks}
    report[f"mrr@{depth}"] = reciprocal_rank_sum / n
    report["n"] = n
    report["misses"] = misses
    return report
