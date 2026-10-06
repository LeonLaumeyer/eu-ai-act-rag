# Retrieval results

| Setting | Hit@1 | Hit@5 (95% CI) | Hit@5 article level | Recall@5 | MRR@10 | ms / question |
|---|---|---|---|---|---|---|
| fixed + dense | 69% | 87% (77%–94%) | 90% | 86% | 0.76 | 141 |
| article + dense | 79% | 96% (90%–100%) | 96% | 90% | 0.86 | 151 |
| article + bm25 | 38% | 65% (52%–77%) | 77% | 60% | 0.50 | 1 |
| article + hybrid | 56% | 85% (73%–94%) | 92% | 78% | 0.68 | 154 |
| article + dense + rerank | 81% | 98% (94%–100%) | 98% | 96% | 0.89 | 13835 |
| article + bm25 + rerank | 71% | 85% (73%–94%) | 92% | 81% | 0.78 | 13572 |
| article + hybrid + rerank | 81% | 96% (90%–100%) | 96% | 94% | 0.88 | 13544 |

Best setting: **article + dense + rerank**.
Paired bootstrap, Hit@5, `fixed + dense` vs `article + dense + rerank`: p = 0.002.

## Hit@5 by language

| Setting | de (n=19) | en (n=33) |
|---|---|---|
| fixed + dense | 79% | 91% |
| article + dense | 100% | 94% |
| article + bm25 | 53% | 73% |
| article + hybrid | 79% | 88% |
| article + dense + rerank | 100% | 97% |
| article + bm25 + rerank | 79% | 88% |
| article + hybrid + rerank | 100% | 94% |

## Hit@5 by question type

| Setting | cross_reference (n=4) | definition (n=7) | factual (n=26) | procedural (n=15) |
|---|---|---|---|---|
| fixed + dense | 100% | 86% | 88% | 80% |
| article + dense | 100% | 100% | 92% | 100% |
| article + bm25 | 100% | 43% | 58% | 80% |
| article + hybrid | 100% | 86% | 77% | 93% |
| article + dense + rerank | 100% | 100% | 96% | 100% |
| article + bm25 + rerank | 100% | 86% | 77% | 93% |
| article + hybrid + rerank | 100% | 100% | 92% | 100% |
