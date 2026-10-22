#!/usr/bin/env bash
# Exercise every endpoint of a running API.
set -euo pipefail
API=${DARIJA_RAG_API:-http://127.0.0.1:8000}

curl -s "$API/api/health"; echo
curl -s "$API/api/search" -H 'content-type: application/json' \
  -d '{"query": "مدة إجازة الولادة", "k": 3, "retriever": "bm25"}'; echo
curl -s "$API/api/ask" -H 'content-type: application/json' \
  -d '{"question": "wach 3andi l7a9 f nhar ra7a f simana?"}'; echo
