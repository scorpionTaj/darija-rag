# Evaluation

Two harnesses share one question file, `eval/questions.jsonl`.

| Command                     | Measures                                                        | Needs LLM                   |
| --------------------------- | --------------------------------------------------------------- | --------------------------- |
| `darija-rag eval`         | retrieval: Recall@1/5/10, MRR@10                                | only for `+rewrite` specs |
| `darija-rag eval-answers` | answers: accuracy, faithfulness, citations, abstention, latency | yes                         |

Both print one row per group: `all`, each `lang=`, and each `type=`.

## Question file

One JSON object per line:

```json
{"id": "q170", "question": "chhal mn sa3a khasni nkhdem f simana?", "lang": "arabizi",
 "type": "factual",
 "relevant": [{"doc": "labour_code_ar.pdf", "page": 59}, {"doc": "labour_code_fr.pdf", "page": 73}],
 "answer": "44 sa3a f simana (lmadda 184)."}
```

| Field        | Meaning                                                                                                                    |
| ------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `lang`     | `msa`, `darija` (Arabic script), `arabizi` (Darija in Latin letters, 3=ع 7=ح 9=ق), `fr`                         |
| `type`     | `factual`, `cross_lingual` (French question, Arabic-only source), `unanswerable`                                     |
| `relevant` | every page that answers it. Labour Code articles count in both the Arabic and French versions. Empty for `unanswerable`. |
| `answer`   | short reference answer with the article number, used by the LLM judge                                                      |

The older `{"doc": ..., "page": ...}` form is still accepted.

### Current set: 200 questions

|               | msa | darija | arabizi | fr | total |
| ------------- | --- | ------ | ------- | -- | ----- |
| factual       | 74  | 44     | 21      | 28 | 167   |
| cross_lingual | –  | –     | –      | 13 | 13    |
| unanswerable  | 6   | 5      | 3       | 6  | 20    |

### Writing rules

1. Write each question from a page you have read. Put the page in `relevant` and the article in
   `answer`.
2. Darija and Arabizi questions use everyday words (السيمانة، الكونجي، الباطرون، جراو عليا,
   `chhal`, `wach`, `lkhedma`), not the vocabulary of the code. That gap is what is being measured.
3. Unanswerable questions should be plausible for a Moroccan user but outside the corpus: tax,
   traffic, criminal law, nationality, company law. Some are traps: "minimum wage in dirhams" sounds
   like the Labour Code, but the amount is set by decree and is not in it.
4. Run `darija-rag eval` after adding questions. A question every retriever misses is often a
   labeling mistake, so check its page before blaming the retriever.

### Known biases

- The MSA questions were written while reading the article and share its wording. That favors BM25,
  so read their scores as an upper bound.
- The Darija detector's word list (`rewrite.py`) was tuned on this set, so it labels all 200
  correctly. Expect lower accuracy on new phrasings.
- When the judge and the generator are the same model, the judge is lenient. Set
  `DARIJA_RAG_JUDGE_MODEL` to a different model for stricter scores.

## Answer metrics

| Metric                | Over                    | Definition                                                          |
| --------------------- | ----------------------- | ------------------------------------------------------------------- |
| Accuracy              | all                     | correct answer, or correct abstention on an out-of-scope question   |
| Correct when answered | in-scope, not abstained | the judge agrees with the reference on the key facts                |
| Faithful              | in-scope, not abstained | the judge finds every claim supported by the given sources          |
| Citation hit          | in-scope, not abstained | at least one cited source is a labeled relevant page                |
| False abstention      | in-scope                | the system said "not in the documents" although the answer is there |
| Abstention recall     | out-of-scope            | the system correctly said "not in the documents"                    |
| p50 / p95 s           | all                     | end-to-end latency, retrieval plus generation                       |

Per-question rows go to `data/processed/answers.jsonl`, including the rewritten query, the answer
and its citations. Read the failures there.
