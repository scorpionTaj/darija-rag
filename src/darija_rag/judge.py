"""Answer-level evaluation.

Deterministic checks (no LLM):
  - abstention: out-of-scope questions must abstain; answerable ones must not
  - citation hit: at least one cited source is a labeled relevant page
LLM-judged checks (answerable, non-abstained answers):
  - correct: agrees with the reference answer on the key facts
  - faithful: every claim is supported by the cited sources (no hallucination)

The judge can be a different model from the generator (DARIJA_RAG_JUDGE_MODEL); using the same
model tends to be lenient with its own mistakes, so treat same-model scores as optimistic.
"""

import json
import re
import statistics

from darija_rag.answer import Answer, language_ok
from darija_rag.llm import LLMClient

_JUDGE = (
    "You grade answers of a Moroccan-law QA system. Inputs: a question, a reference answer, the "
    "system answer and the sources it was given. Reply with JSON only: "
    '{"correct": true|false, "faithful": true|false}. '
    "correct: the system answer states the key facts of the reference (numbers, durations, "
    "conditions); wording and language may differ, extra correct detail is fine. "
    "faithful: every claim in the system answer is supported by the sources."
)


def judge_answer(llm: LLMClient, question: dict, answer: Answer) -> dict:
    # Only the cited sources: faithfulness is about what the answer claims to rest on, and a
    # shorter prompt roughly halves judge time. Uncited answers are checked against everything.
    cited = answer.cited or list(range(len(answer.sources)))
    sources = "\n\n".join(
        f"[{i + 1}] ({c.doc}, page {c.page}) {c.text}" for i in cited for c in [answer.sources[i]]
    )
    user = (
        f"Question: {question['question']}\nReference answer: {question['answer']}\n"
        f"System answer: {answer.text}\n\nSources:\n{sources}"
    )
    reply = llm.chat(
        [{"role": "system", "content": _JUDGE}, {"role": "user", "content": user}], max_tokens=60
    )
    return parse_verdict(reply)


def parse_verdict(reply: str) -> dict:
    match = re.search(r"\{.*?\}", reply, flags=re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    return {"correct": data.get("correct") is True, "faithful": data.get("faithful") is True}


def score(question: dict, answer: Answer, verdict: dict | None) -> dict:
    """One result row; verdict is None when the LLM judge was not run."""
    targets = {(r["doc"], r["page"]) for r in question["relevant"]}
    in_scope = bool(targets)
    row = {
        "id": question["id"],
        "lang": question.get("lang"),
        "type": question.get("type"),
        "in_scope": in_scope,
        "abstained": answer.abstained,
        "abstention_ok": answer.abstained != in_scope,
        "citation_hit": any((c.doc, c.page) in targets for c in answer.cited_chunks),
        "language_match": answer.abstained or language_ok(answer.text, answer.variety),
        "language_fixed": answer.language_fixed,
        "total_s": answer.timings.get("total_s", 0.0),
        "question": question["question"],
        "retrieval_query": answer.retrieval_query,
        "answer": answer.text,
        "cited": [f"{c.doc}#p{c.page}" for c in answer.cited_chunks],
    }
    if in_scope and not answer.abstained and verdict is not None:
        row.update(verdict)
    elif not in_scope:
        row.update({"correct": answer.abstained, "faithful": True})
    else:
        row.update({"correct": False, "faithful": True})  # abstained on an answerable question
    return row


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    in_scope = [r for r in rows if r["in_scope"]]
    out_scope = [r for r in rows if not r["in_scope"]]
    answered = [r for r in in_scope if not r["abstained"]]
    latencies = sorted(r["total_s"] for r in rows)

    def rate(items, key):
        return sum(bool(r.get(key)) for r in items) / len(items) if items else None

    return {
        "n": len(rows),
        "accuracy": rate(rows, "correct"),
        "answered_correct": rate(answered, "correct"),
        "faithfulness": rate(answered, "faithful"),
        "citation_hit": rate(answered, "citation_hit"),
        "language_match": rate(answered, "language_match"),
        "false_abstention": rate(in_scope, "abstained"),
        "abstention_recall": rate(out_scope, "abstained"),
        "p50_s": statistics.median(latencies),
        "p95_s": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))],
    }


def answer_table(rows: list[dict]) -> str:
    def fmt(value, digits=2):
        return "–" if value is None else f"{value:.{digits}f}"

    from darija_rag.evaluate import group

    header = (
        "| Questions | n | Accuracy | Correct when answered | Faithful | Citation hit "
        "| Language match | False abstention | Abstention recall | p50 s | p95 s |"
    )
    lines = [header, "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, subset in group(rows).items():
        s = summarize(subset)
        lines.append(
            f"| {name} | {s['n']} | {fmt(s['accuracy'])} | {fmt(s['answered_correct'])} "
            f"| {fmt(s['faithfulness'])} | {fmt(s['citation_hit'])} | {fmt(s['language_match'])} "
            f"| {fmt(s['false_abstention'])} "
            f"| {fmt(s['abstention_recall'])} | {fmt(s['p50_s'], 1)} | {fmt(s['p95_s'], 1)} |"
        )
    return "\n".join(lines)
