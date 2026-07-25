# Dataset

This project uses the **Amazon Reviews 2023** dataset — All_Beauty category.

**Citation:** Hou et al. (2024). Bridging Language and Items for Retrieval and Recommendation. arXiv:2403.03952.

---

## Download Instructions

The raw review file is ~300MB. Download it directly from the McAuley Lab:

```bash
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz -P data/
```

Or via curl:
```bash
curl -o data/All_Beauty.jsonl.gz \
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz
```

Place the file at `data/All_Beauty.jsonl.gz`. The notebook reads from this path.

---

## What the Notebook Does With It

| Step                        | Output              |
|-----------------------------|---------------------|
| Scans 701,528 records       | Full review stream  |
| Seeded reservoir sample (seed=42) | 50,000 reviews |
| Min-length filter (≥5 tokens) | 43,294 reviews  |
| Deduplication on cleaned text | 43,181 reviews  |

The final corpus of **43,181 review documents** across **23,512 unique ASINs** is used for all retrieval experiments.

---

## License

License not specified on the McAuley Lab project page or Hugging Face dataset card as of June 2026. Check [https://amazon-reviews-2023.github.io/](https://amazon-reviews-2023.github.io/) before use in commercial applications.
