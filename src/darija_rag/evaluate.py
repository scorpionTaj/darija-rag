"""Retrieval metrics against a labeled question set.

Each line of the question file is JSON:
    {"id": "q001", "question": "...", "lang": "darija", "type": "factual",
     "relevant": [{"doc": "labour_code_ar.pdf", "page": 59}, {"doc": "labour_code_fr.pdf", "page": 73}],
     "answer": "reference answer used by the answer judge"}

A retrieved chunk is relevant when its (doc, page) is in "relevant"; several pages can be relevant
(the Arabic and French Labour Code carry the same articles). Questions with an empty "relevant" list
are out of scope: retrieval metrics skip them, and the answer eval expects an abstention.
The older {"doc": ..., "page": ...} form is still accepted.
"""

import json
import random
from collections.abc import Callable
from pathlib import Path

from darija_rag.ingest import Chunk

Retrieve = Callable[[str, int], list[tuple[Chunk, float]]]


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    for number, q in enumerate(questions, start=1):
        where = f"{path}:{number}"
        if "question" not in q:
            raise ValueError(f"{where} missing field 'question'")
        if "relevant" not in q:
            if not {"doc", "page"} <= q.keys():
                raise ValueError(f"{where} needs 'relevant' or 'doc' + 'page'")
            q["relevant"] = [{"doc": q.pop("doc"), "page": q.pop("page")}]
        for ref in q["relevant"]:
            # type() rather than isinstance: a bool is an int and must be rejected.
            if type(ref.get("doc")) is not str or type(ref.get("page")) is not int:
                raise ValueError(f"{where} bad relevant entry {ref!r}")
        q.setdefault("id", f"line{number}")
        q.setdefault("type", "factual" if q["relevant"] else "unanswerable")
    return questions


def answerable(questions: list[dict]) -> list[dict]:
    return [q for q in questions if q["relevant"]]


def group(questions: list[dict]) -> dict[str, list[dict]]:
    """'all', then one group per language and per question type."""
    groups = {"all": questions}
    for field in ("lang", "type"):
        for value in sorted({q[field] for q in questions if field in q}):
            groups[f"{field}={value}"] = [q for q in questions if q.get(field) == value]
    return groups


def stratified_sample(questions: list[dict], n: int, seed: int = 0) -> list[dict]:
    """About n questions with every (lang, type) cell represented in proportion, at least one
    each, so a quick run still covers Arabizi and out-of-scope questions. Keeps file order."""
    cells: dict[tuple, list[dict]] = {}
    for q in questions:
        cells.setdefault((q.get("lang"), q.get("type")), []).append(q)
    rng = random.Random(seed)
    picked = set()
    for members in cells.values():
        share = max(1, round(n * len(members) / len(questions)))
        picked.update(q["id"] for q in rng.sample(members, min(share, len(members))))
    return [q for q in questions if q["id"] in picked]


def first_relevant_rank(q: dict, results: list[tuple[Chunk, float]]) -> int | None:
    targets = {(ref["doc"], ref["page"]) for ref in q["relevant"]}
    for position, (chunk, _) in enumerate(results, start=1):
        if (chunk.doc, chunk.page) in targets:
            return position
    return None


def ranks(retrieve: Retrieve, questions: list[dict], depth: int = 10) -> dict[str, int | None]:
    """Rank of the first relevant chunk for each answerable question (None = not in top depth)."""
    return {
        q["id"]: first_relevant_rank(q, retrieve(q["question"], depth))
        for q in answerable(questions)
    }


def metrics(question_ranks: dict[str, int | None], ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    if not question_ranks:
        raise ValueError("no answerable questions to evaluate")
    n = len(question_ranks)
    found = [r for r in question_ranks.values() if r is not None]
    report = {f"recall@{k}": sum(r <= k for r in found) / n for k in ks}
    report[f"mrr@{max(ks)}"] = sum(1 / r for r in found if r <= max(ks)) / n
    report["n"] = n
    report["misses"] = [qid for qid, r in question_ranks.items() if r is None]
    return report


def evaluate(retrieve: Retrieve, questions: list[dict], ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    return metrics(ranks(retrieve, questions, max(ks)), ks)
