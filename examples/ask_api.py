"""Ask the running API a question and print the answer with its sources.

python examples/ask_api.py "chhal katdom periode d'essai f lkhedma?"
DARIJA_RAG_API=http://localhost:8000 python examples/ask_api.py "..."
"""

import os
import sys

import httpx

API = os.environ.get("DARIJA_RAG_API", "http://127.0.0.1:8000")


def main() -> None:
    question = " ".join(sys.argv[1:]) or "ما هي مدة إجازة الولادة للأجيرة؟"
    response = httpx.post(f"{API}/api/ask", json={"question": question}, timeout=300)
    if response.status_code != 200:
        sys.exit(f"{response.status_code}: {response.json().get('detail')}")
    data = response.json()
    if data["retrieval_query"] != question:
        print(f"(searched as: {data['retrieval_query']})")
    print(data["answer"])
    for c in data["citations"]:
        print(f"  [{c['n']}] {c['doc']} p.{c['page']}")
    print(f"({data['timings']['total_s']}s, {data['variety']}, {data['model']})")


if __name__ == "__main__":
    main()
