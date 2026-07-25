# RAG System for E-Commerce Product Discovery

A retrieval-augmented generation pipeline that answers natural-language product queries using customer review evidence. Built and evaluated on 43,181 Amazon Beauty reviews.

> **MSc Thesis** — GISMA Business School Berlin, 2026  
> Supervisor: Dr. Hiba Khalid

---

## What This Does

Users often search for products using conditions, not keywords:  
*"fragrance-free cleanser for sensitive skin"* or *"hydrating serum for dry skin"*

Standard keyword search struggles here. This pipeline retrieves relevant customer reviews using a hybrid of lexical + semantic search, then generates a grounded answer using a language model — citing specific reviews as evidence.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           Hybrid Retriever              │
│                                         │
│   BM25 (lexical)  +  FAISS (semantic)  │
│         ↓                  ↓            │
│      Min-max score fusion (λ=0.5)      │
│                  ↓                      │
│        Top-K ranked reviews             │
└─────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│         RAG Generator                   │
│   microsoft/Phi-3-mini-4k-instruct      │
│   Conditioned on top-5 reviews          │
│   Deterministic decoding               │
└─────────────────────────────────────────┘
                   │
                   ▼
         Grounded answer + cited review IDs
```

---

## Key Results

Evaluated across 40 product-discovery queries with manually labeled relevance judgments (1,746 judged query-document pairs, Holm-Bonferroni corrected significance tests).

### Retrieval Performance at k=10

| Method            | Precision@10 | Recall@10 | MRR@10 | nDCG@10 | Latency (s) |
|-------------------|-------------|-----------|--------|---------|-------------|
| **Hybrid** ✓     | **0.8650**  | **0.2687**| **0.9750** | **0.7576** | 0.0771 |
| BM25              | 0.8050      | 0.2487    | 0.9313 | 0.6911  | 0.0554      |
| FAISS (dense)     | 0.8275      | 0.2555    | 0.9675 | 0.6826  | 0.0115      |
| Field-weighted BM25 | 0.8225   | 0.2527    | 0.9083 | 0.6733  | 0.0965      |

Hybrid significantly outperformed FAISS and field-weighted BM25 on nDCG@10 after Holm correction. Comparison with standard BM25 was not significant — BM25 remains a strong baseline.

### RAG vs No-Retrieval Generation (20 queries, Phi-3-mini)

| Criterion       | RAG Hybrid | No Retrieval | Significant? |
|-----------------|-----------|-------------|--------------|
| Relevance       | 4.50      | 3.15        | ✓ Yes        |
| Groundedness    | 2.85      | 1.00        | ✓ Yes        |
| Helpfulness     | 3.65      | 2.60        | ✓ Yes        |
| Clarity         | 2.20      | 1.85        | ✗ No         |
| Hallucination ↓ | 3.05      | 3.30        | ✗ No         |

RAG significantly improved relevance, groundedness, and helpfulness. Clarity and hallucination-risk reduction were not significant — honest limitations documented in the thesis.

---

## Dataset

Amazon Reviews 2023 — All_Beauty category (McAuley Lab, UCSD)

| Step                     | Count   |
|--------------------------|---------|
| Raw records scanned      | 701,528 |
| Seeded reservoir sample  | 50,000  |
| After length filtering   | 43,294  |
| After deduplication      | 43,181  |
| Unique product ASINs     | 23,512  |

See `data/README.md` for download instructions.

---

## Retrieval Methods Compared

**BM25** — Lexical ranking via `rank_bm25`. Strong on exact constraints like *"waterproof"*, *"alcohol-free"*.

**FAISS Dense** — Sentence embeddings via `all-MiniLM-L6-v2`, indexed with `IndexFlatIP`. Captures semantic paraphrase. Fastest latency (0.0115s).

**Field-weighted BM25** — Separate BM25 scores for review title (w=0.30) and review body (w=0.70), combined after min-max normalisation.

**Hybrid** ← best overall — Union of top-100 BM25 + top-100 FAISS candidates, min-max normalised and fused with λ=0.5.

---

## How To Run

### 1. Clone and install
```bash
git clone https://github.com/kushalaralpati/Design-and-evaluation-of-RAG-system-for-product-discovery-in-ecommerce
cd Design-and-evaluation-of-RAG-system-for-product-discovery-in-ecommerce
pip install -r requirements.txt
```

### 2. Download the dataset
Follow the instructions in `data/README.md` to download `All_Beauty.jsonl.gz`.

### 3. Run the notebook
```bash
jupyter notebook notebooks/rag_pipeline.ipynb
```
Or open directly in Google Colab (GPU recommended for embedding generation).

---

## Evaluation Design

- **40 researcher-designed queries** across 11 intent categories (skin concern, hair concern, restriction, ethical preference, multi-condition, etc.)
- **Relevance judgments** pooled at depth 20 across all 4 retrieval methods → 1,746 unique (query, doc) pairs
- **3-level relevance scale**: 0 = not relevant, 1 = partially relevant, 2 = highly relevant
- **Statistical validation**: query-level bootstrap CIs (2,000 iterations), paired sign-randomisation tests (5,000 iterations), Holm-Bonferroni correction across 12 retrieval comparisons
- **RAG evaluation**: blinded human scoring on 5 criteria across 20 queries; 10,000 bootstrap + 50,000 sign-randomisation iterations, Holm correction

---

## Limitations (documented, not hidden)

- Single evaluator for relevance labels and answer scoring — no inter-rater reliability reported
- Researcher-designed query set, not from real search logs
- One product category (All_Beauty), one review dataset release
- Phi-3-mini with deterministic decoding, one run per query
- Latency measured once per call in Colab — not a hardware-independent benchmark
- Only 3 hybrid weight values tested (λ = 0.2, 0.5, 0.8)

---

## Tech Stack

`Python` · `sentence-transformers` · `FAISS` · `rank-bm25` · `Hugging Face Transformers` · `Phi-3-mini-4k-instruct` · `scikit-learn` · `pandas` · `matplotlib`

---

## Thesis

Full thesis PDF available in this repository: [`RAG_Product_Discovery_Thesis.pdf`](./RAG_Product_Discovery_Thesis.pdf)

---

## Citation

```bibtex
@mastersthesis{srinivas2026rag,
  author    = {Kushala Ralpati Srinivas},
  title     = {Design and Evaluation of a Retrieval-Augmented Generation System for Product Discovery in E-Commerce},
  school    = {GISMA Business School Berlin},
  year      = {2026},
  month     = {June}
}
```

---

## References

- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
- Reimers & Gurevych (2019). Sentence-BERT. EMNLP.
- Johnson et al. (2021). Billion-Scale Similarity Search with GPUs. IEEE Transactions on Big Data.
- Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
- Hou et al. (2024). Bridging Language and Items for Retrieval and Recommendation. arXiv:2403.03952.
