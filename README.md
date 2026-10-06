# EU AI Act RAG

A question-answering system over the EU AI Act (Regulation (EU) 2024/1689) in English and German. It answers with citations such as `[Art. 6(2)]`, checks those citations against the text it retrieved, and comes with an evaluation that compares chunking and retrieval settings on a set of 60 questions.

I built this to learn the technologies commonly used in RAG systems and to measure what each one is good at, where it fails and how they compare.

## How It Works

```
EUR-Lex HTML (EN, DE)
      │  parse: one unit per article paragraph, definition, annex point, recital
      ▼
808 units per language, each with a citation ("Art. 6(2)" / "Art. 6 Abs. 2")
      │  chunk: one chunk per unit (long units split, with the lead-in repeated)
      ▼
┌──────────────────────┐      ┌───────────────────────┐
│ dense: bge-m3 vectors│      │ keyword: BM25 with    │
│ in Qdrant            │      │ stemming per language │
└──────────┬───────────┘      └───────────┬───────────┘
           └────── RRF fusion (hybrid) ───┘
                        │  optional: cross-encoder reranker
                        ▼
              top 5 excerpts, labelled with their citation
                        │
                        ▼
      LLM (Ollama, Claude or Gemini, set in config.yaml)
                        │
                        ▼
   answer + citation check (every [..] must be one of the excerpts)
```

- **Structure-aware parsing.** The official HTML marks articles and paragraphs, so every unit has an exact citation and the same id in both languages (`art6-2`). One gold label therefore works for English and German.
- **Two chunking strategies.** `article` follows the structure of the law. `fixed` is the usual baseline: windows of 512 tokens with 64 tokens overlap, ignoring structure. Each fixed chunk records which units it covers, so both strategies can be scored against the same gold labels.
- **Three retrieval modes.** `dense`, `bm25` and `hybrid` (reciprocal rank fusion), each with or without a multilingual cross-encoder reranker (`bge-reranker-v2-m3`).
- **Checked citations.** The LLM must cite the label of the excerpt it used. A citation that is not among the excerpts it was shown is reported as invalid. Citing a point inside an excerpt, such as `[Art. 5(1)(d)]` for the excerpt `Art. 5(1)`, counts as valid.
- **Refusal.** If the excerpts don't contain the answer, the LLM must reply with one fixed sentence, which makes refusals measurable.

## Results: Retrieval

52 answerable questions (33 English, 19 German). A question counts as a hit when one of the first five results covers a gold unit. "Unit level" means the exact paragraph; "article level" means the right article or annex.

| Setting | Hit@1 | Hit@5 (95% CI) | Hit@5 article level | Recall@5 | MRR@10 | ms / question |
|---|---|---|---|---|---|---|
| fixed + dense | 69% | 87% (77%–94%) | 90% | 86% | 0.76 | 141 |
| article + dense | 79% | 96% (90%–100%) | 96% | 90% | 0.86 | 151 |
| article + bm25 | 38% | 65% (52%–77%) | 77% | 60% | 0.50 | 1 |
| article + hybrid | 56% | 85% (73%–94%) | 92% | 78% | 0.68 | 154 |
| article + dense + rerank | 81% | **98%** (94%–100%) | 98% | 96% | 0.89 | 13,835 |
| article + bm25 + rerank | 71% | 85% (73%–94%) | 92% | 81% | 0.78 | 13,572 |
| article + hybrid + rerank | 81% | 96% (90%–100%) | 96% | 94% | 0.88 | 13,544 |

Times are on a CPU without a GPU. Without the reranker almost all of it is embedding the question; with it, almost all of it is the cross-encoder scoring 30 candidates.

### What the Numbers Say

- **Article-based chunking beats fixed-size chunking:** Hit@5 rises from 87% to 96% and Hit@1 from 69% to 79% with the same embedding model. A paired bootstrap over the questions gives p = 0.03 for that Hit@5 difference, and p = 0.002 for the baseline against the best setting. The intervals still overlap, because 52 questions are not many.
- **Hybrid retrieval did not help here; it hurt.** Dense search alone reaches 96%, BM25 alone 65%, and fusing them gives 85%. Reciprocal rank fusion weighs both rankings equally, so a weak ranking pulls a strong one down. On 6 questions, dense search finds the gold unit and the fused list loses it.
- **The reranker repairs weak candidate lists.** It lifts BM25 from 65% to 85% Hit@5 and from 38% to 71% Hit@1, and hybrid from 85% back to 96%. As long as the right chunk is somewhere among the 30 candidates, the cross-encoder usually moves it up.
- **On top of dense search the reranker adds little, at a high price.** Hit@5 goes from 96% to 98% (one more question) and Hit@1 from 79% to 81%: it moves the gold unit to first place on 4 questions and away from first place on 3. That costs about 14 seconds per question on this CPU, roughly 90 times the latency. Recall@5 gains more (90% to 96%), which matters for questions whose answer is spread over two places.
- **BM25 is weak on these questions by design.** The questions were written in everyday words, not in the wording of the Act ("banned" instead of "prohibited", "hobby" instead of "purely personal non-professional activity"). That is how real users ask, and it is where keyword search struggles. BM25 still ranks the gold unit first on 5 questions where dense search does not, mostly ones with distinctive terms.
- **German keyword search is worse than English:** 53% against 73% Hit@5. German compounds such as *Hochrisiko-KI-System* or *Marktüberwachungsbehörde* don't match their parts after stemming. Dense search shows no such gap (100% German, 94% English).
- **Definitions are the hardest type for BM25** (43%): the 68 definitions in Art. 3 are short and share most of their vocabulary.

Hit@5 by language and question type:

| Setting | de (n=19) | en (n=33) | cross-reference (n=4) | definition (n=7) | factual (n=26) | procedural (n=15) |
|---|---|---|---|---|---|---|
| fixed + dense | 79% | 91% | 100% | 86% | 88% | 80% |
| article + dense | 100% | 94% | 100% | 100% | 92% | 100% |
| article + bm25 | 53% | 73% | 100% | 43% | 58% | 80% |
| article + hybrid | 79% | 88% | 100% | 86% | 77% | 93% |
| article + dense + rerank | 100% | 97% | 100% | 100% | 96% | 100% |
| article + bm25 + rerank | 79% | 88% | 100% | 86% | 77% | 93% |
| article + hybrid + rerank | 100% | 94% | 100% | 100% | 92% | 100% |

### Where It Still Fails

The best setting (`article + dense + rerank`) misses 1 of 52 questions:

- **"When do the obligations for general-purpose AI models start to apply?"** The dates are in Art. 113. The retriever returns the obligations themselves (Art. 53, 55) and the transition rules for existing systems and models (Art. 111), which are about the right topic but don't contain the general start date. The reranker does not help, because Art. 113 is not among its candidates.

Without the reranker, dense search also misses a second question:

- **"Am I bound by the Act if I use an AI tool privately, as a hobby?"** The answer is Art. 2(10), a single sentence of 25 words about "purely personal non-professional activity". The question shares no key term with it, and longer paragraphs about deployers rank higher. The reranker moves Art. 2(10) to first place.

Two more weaknesses showed up outside the hit rate:

- **Long paragraphs are shown only in part.** Art. 5(1), the list of prohibited practices, is split into several chunks. Duplicate pieces of one unit are removed from the results, so the LLM sees only one piece and its answer to "which practices are banned" lists some of the practices, not all.
- **A hit at article level is not always enough.** In the fixed-size baseline, 3 points of Hit@5 separate unit level from article level: the right article was found, but not the paragraph that answers the question.

The full list of misses with the retrieved citations is in `results/failures.md`.

## Results: Answers

All 60 questions through the full pipeline. A small local model (`gemma-4-e2b` via Ollama) writes the answers, once with dense retrieval alone and once with the reranker added.

| Measure | article + dense | article + dense + rerank |
|---|---|---|
| Unanswerable questions refused (8) | 88% | 100% |
| Answerable questions refused by mistake (52) | 4% | 0% |
| Answers with at least one citation | 100% | 100% |
| Answers whose citations are all valid | 100% | 100% |
| Answers that cite a gold unit | 98% | 98% |
| Faithfulness (LLM judge) | not judged | 99.7% |
| Seconds per question | 4.6 | 19.5 |

The citation rows look at the answerable questions that got a real answer (50 without the reranker, 52 with it).

- **No invented citations.** Every citation in every answer points to an excerpt the model was shown.
- **Better retrieval fixed all three refusal errors.** Without the reranker the model wrongly refused two answerable questions and answered one unanswerable one. With it, the "hobby" question gets Art. 2(10) and is answered correctly, the German question about real-time face recognition is answered from Art. 5(2), and the false-premise question "Which AI systems does Article 6 prohibit?" is refused. The LLM is the same in both runs; only the excerpts changed.
- **A cited answer can still be wrong.** For "When do the obligations for general-purpose AI models start to apply?", retrieval misses Art. 113 in both runs. The model answers from Art. 111(3) with the deadline for models already on the market (2 August 2027) instead of the general start date. The citation is valid, the sentence matches its source, and the judge rates it fully faithful; it just answers a different question. Citation checks and faithfulness measure grounding, not correctness.

### Faithfulness

A second LLM (`gemini-3.1-flash-lite`) splits each answer into claims and marks each claim as supported by the excerpts or not. Refusals are not judged. Of the 52 judged answers, 51 were rated fully supported and one at 83% (5 of 6 claims), which gives the mean of 99.7%.

I checked 6 of the 52 judged answers by hand and agreed with the judge on all of them. The answers read like lightly reworded excerpts, which explains the high score: the model rarely adds anything of its own. Six answers that were all correct can't show whether the judge would catch an unsupported claim, so the score is best read as "the model stays close to its sources", not as proof that the judge is reliable.

## The Question Set

`eval/questions.yaml` holds 60 questions:

- 38 English and 22 German, including 10 translated pairs (the same question in both languages).
- 52 answerable questions with gold units: 26 factual, 15 procedural, 7 definitions, 4 that need two places in the Act (an article and an annex point).
- 8 questions the Act does not answer: out of scope ("fines for GDPR violations"), a false premise ("Which AI systems does Article 6 prohibit?"), or plausible but absent ("minimum salary of an AI compliance officer").

The questions were written and their gold labels checked against the parsed text before the first retrieval run, and they were not changed afterwards. Loading the file fails if a gold label does not exist in the parsed Act.

Limits of this set: it is small, so the confidence intervals are wide; the gold labels come from one annotator; and the questions were written by the same person who built the system.

## Quickstart

Requires Python 3.11 or newer. For answers you also need [Ollama](https://ollama.com) with a model pulled, or an API key for Claude or Gemini. Keys go into a `.env` file in the project root (see `.env.example`) or into the environment variables `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`.

```bash
python -m venv venv
venv\Scripts\activate
pip install -e ".[dev]"
```

Build the data (the parsed and chunked text is created from the HTML in `data/raw/`):

```bash
python -m aiact_rag.ingest.parse
python -m aiact_rag.ingest.chunk
```

Start the API. The first start embeds all chunks (a few minutes on a CPU) and caches the vectors in `data/embeddings/`:

```bash
uvicorn aiact_rag.main:create --factory
```

Ask a question:

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"Which uses of AI are completely banned?\"}"
```

The response contains `answer`, `lang` (detected if not given), `refused`, `citations` (valid), `invalid_citations`, and `sources` with the citation, text and score of each excerpt.

### Configuration

`config.yaml` selects the chunking (`article` or `fixed`), the retrieval mode (`dense`, `bm25` or `hybrid`), the reranker (on or off) and the LLM (`ollama`, `anthropic` or `gemini`). Invalid values fail at start with a message that names the key.

For Gemini, `min_interval` sets the least number of seconds between two requests, to stay under a requests-per-minute limit (`4` for 15 requests per minute). A rate limit (HTTP 429) and a temporary server error (500, 503) are retried up to five times; the wait after a server error doubles each time.

## Running the Evaluation

```bash
python -m aiact_rag.retrieval_eval                # all seven settings
python -m aiact_rag.retrieval_eval --no-rerank    # the four settings without the reranker
python -m aiact_rag.answer_eval --mode dense --no-rerank
python -m aiact_rag.answer_eval --judge gemini    # adds faithfulness, judged by a second LLM
python -m aiact_rag.answer_eval --judge gemini --resume   # continue a run that stopped
```

Results are written to `results/`: the tables as Markdown and one JSON line per question, so single cases can be inspected without running everything again.

The answer evaluation writes each answer as soon as it is there. If a run stops halfway (the LLM is down, a daily limit is reached), `--resume` keeps the answers already written and asks only the remaining questions. It refuses to continue a run that was made with another model, judge or retrieval setting (recorded in `answers_<provider>.run.json`), so one table never mixes two settings.

## Tests

```bash
pytest              # 391 tests, no model or server needed
pytest -m slow      # the tests that load the real embedding model and reranker
```

The tests use fake embeddings, a fake LLM and an in-memory Qdrant, so they run in seconds. HTTP calls to Ollama are tested with a mock transport.

## Project Layout

```
src/aiact_rag/
  ingest/parse.py, chunk.py     HTML → units → chunks
  index/embed.py                bge-m3 embeddings with a cache on disk
  index/vector_store.py         Qdrant with language and kind filters
  index/bm25.py                 BM25 with stemming per language
  retrieve/hybrid.py, rerank.py RRF fusion, deduplication, reranking
  generate.py                   prompt, citation check, refusal
  llm/                          Ollama, Claude and Gemini behind one interface
  api.py, main.py               FastAPI app and entry point
  config.py, pipeline.py        settings and wiring
  evaluate.py                   metrics and bootstrap statistics
  retrieval_eval.py             the retrieval comparison
  judge.py, answer_eval.py      answer evaluation
eval/questions.yaml             the question set
results/                        evaluation output
tests/
```

## Limitations and Next Steps

- This is a study project and not legal advice. It works on the text as published in the Official Journal and does not track later corrections or guidance.
- Weighted fusion, or using BM25 only for queries with exact terms (article numbers, defined terms), should be tried before concluding that keyword search adds nothing.
- Showing all pieces of a split paragraph to the LLM would fix the incomplete answer on Art. 5(1).
- Splitting German compounds before stemming would likely close part of the gap between German and English keyword search.
- A larger question set with a second annotator would narrow the intervals and make the gold labels more reliable.
- Cross-reference questions are thin: only 4 of 60. Since multi-hop retrieval is where RAG systems usually fail, adding 3–5 more would help.

## Source

The text of the Regulation comes from EUR-Lex (CELEX 32024R1689), © European Union. Reuse is permitted with attribution.
