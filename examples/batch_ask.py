"""Answer every question in a text file (one per line) and write a CSV for review.

Useful for checking a list of real user questions, an FAQ, or a client's inbox before a demo.

    python examples/batch_ask.py questions.txt answers.csv
"""

import csv
import os
import sys
from pathlib import Path

import httpx

API = os.environ.get("DARIJA_RAG_API", "http://127.0.0.1:8000")


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: batch_ask.py QUESTIONS.txt OUT.csv")
    lines = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    questions = [line.strip() for line in lines if line.strip()]
    with (
        httpx.Client(base_url=API, timeout=300) as client,
        open(
            sys.argv[2],
            "w",
            newline="",
            encoding="utf-8-sig",  # BOM so Excel shows Arabic correctly
        ) as out,
    ):
        writer = csv.writer(out)
        writer.writerow(["question", "variety", "abstained", "answer", "sources", "seconds"])
        for number, question in enumerate(questions, start=1):
            response = client.post("/api/ask", json={"question": question})
            if response.status_code != 200:
                writer.writerow([question, "", "", f"ERROR {response.status_code}", "", ""])
                continue
            d = response.json()
            sources = "; ".join(f"{c['doc']} p.{c['page']}" for c in d["citations"])
            writer.writerow(
                [
                    question,
                    d["variety"],
                    d["abstained"],
                    d["answer"],
                    sources,
                    d["timings"]["total_s"],
                ]
            )
            out.flush()
            print(f"{number}/{len(questions)}", file=sys.stderr)


if __name__ == "__main__":
    main()
