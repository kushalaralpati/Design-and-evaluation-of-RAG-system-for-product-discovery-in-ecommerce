import time
import numpy as np
import pandas as pd
import streamlit as st
import faiss
import anthropic
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Product Discovery",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    padding: 2.5rem 2rem 2rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
    color: white;
}
.hero h1  { font-size: 2rem; font-weight: 700; margin: 0 0 0.3rem 0; }
.hero p   { font-size: 0.95rem; opacity: 0.8; margin: 0.2rem 0; }
.hero .sub{ font-size: 0.78rem; opacity: 0.55; margin-top: 0.6rem; }

.review-card {
    border-radius: 10px; padding: 0.9rem 1rem;
    margin: 0.5rem 0; border-left: 4px solid #ccc;
    background: #fafafa;
}
.review-card .brand-tag {
    display: inline-block;
    background: #ede9fe; color: #5b21b6;
    font-size: 0.7rem; font-weight: 600;
    padding: 2px 8px; border-radius: 99px;
    margin-bottom: 5px;
}
.review-card .card-title { font-weight: 600; font-size: 0.88rem; margin-bottom: 4px; }
.review-card .card-text  { font-size: 0.82rem; color: #444; line-height: 1.5; }
.review-card .card-meta  { font-size: 0.7rem; color: #999; margin-top: 6px; }

.rag-box {
    background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
    border: 1.5px solid #86efac;
    border-radius: 12px; padding: 1.4rem 1.6rem;
    font-size: 0.92rem; line-height: 1.8; color: #1a2e1a;
    margin-bottom: 1rem;
}

.brand-pill {
    display: inline-block;
    border-radius: 99px; padding: 4px 12px;
    font-size: 0.78rem; font-weight: 600;
    margin: 3px; border: 1.5px solid;
}
.brand-top    { background:#f0fdf4; color:#166534; border-color:#86efac; }
.brand-mid    { background:#fffbeb; color:#92400e; border-color:#fcd34d; }
.brand-low    { background:#fef2f2; color:#991b1b; border-color:#fca5a5; }

.section-label {
    font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 0.5rem;
}
.footer-bar {
    text-align: center; color: #94a3b8;
    font-size: 0.75rem; padding: 1.2rem 0 0.5rem 0;
    border-top: 1px solid #e2e8f0; margin-top: 1.5rem;
}
.footer-bar a { color: #7c3aed; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DEMO_SAMPLE  = 5000
SEED         = 42
TOP_K        = 5
CANDIDATE_K  = 100

EXAMPLE_QUERIES = [
    "best moisturizer for dry skin",
    "fragrance free cleanser for sensitive skin",
    "waterproof mascara",
    "shampoo for thinning hair",
    "sunscreen without white residue",
    "gentle face wash for acne prone skin",
    "anti aging cream for wrinkles",
    "hydrating serum for dry skin",
    "cruelty free mascara",
    "hair oil for frizzy hair",
]


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    try:
        df = pd.read_csv("enriched_beauty_reviews.csv.gz", compression="gzip")
    except Exception:
        df = pd.read_csv("prepared_beauty_reviews.csv")

    df = df.dropna(subset=["clean_text"]).reset_index(drop=True)
    df["clean_text"]    = df["clean_text"].astype(str)
    df["original_text"] = df.get("original_text", df["clean_text"]).fillna(df["clean_text"]).astype(str)
    df["title"]         = df.get("title", pd.Series(["Review"]*len(df))).fillna("Review").astype(str)
    df["doc_id"]        = df["doc_id"].astype(str)
    df["brand"]         = df.get("brand", pd.Series([""]*len(df))).fillna("").astype(str)
    df["product_title"] = df.get("product_title", pd.Series([""]*len(df))).fillna("").astype(str)

    # Enriched search text: brand + product title + features + review
    if "enriched_text" in df.columns:
        df["enriched_text"] = df["enriched_text"].fillna(df["clean_text"]).astype(str)
    else:
        df["enriched_text"] = (
            df["brand"] + " " + df["product_title"] + " " + df["clean_text"]
        ).str.strip()

    return df.sample(n=min(DEMO_SAMPLE, len(df)), random_state=SEED).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def build_indices(_df):
    tokenized = [t.split() for t in _df["enriched_text"]]
    bm25 = BM25Okapi(tokenized)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs  = model.encode(
        _df["enriched_text"].tolist(),
        batch_size=128,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return bm25, model, index


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve_bm25(query, bm25, df, k=TOP_K):
    t0     = time.perf_counter()
    scores = bm25.get_scores(query.lower().split())
    idx    = np.argsort(scores)[::-1][:k]
    lat    = time.perf_counter() - t0
    res    = df.iloc[idx].copy()
    res["score"] = scores[idx]
    res["rank"]  = range(1, k + 1)
    return res.reset_index(drop=True), lat


def retrieve_hybrid(query, bm25, model, faiss_index, df, k=TOP_K, lam=0.5):
    t0          = time.perf_counter()
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_top    = set(np.argsort(bm25_scores)[::-1][:CANDIDATE_K].tolist())

    q_emb             = model.encode([query], normalize_embeddings=True).astype("float32")
    d_scores_raw, d_idx = faiss_index.search(q_emb, CANDIDATE_K)
    dense_top  = set(d_idx[0].tolist())
    dense_map  = {int(i): float(s) for i, s in zip(d_idx[0], d_scores_raw[0])}

    pool = list(bm25_top | dense_top)
    pb   = np.array([bm25_scores[i] for i in pool])
    pd_  = np.array([dense_map.get(i, 0.0) for i in pool])

    def mm(a):
        lo, hi = a.min(), a.max()
        return np.ones_like(a) if hi == lo else (a - lo) / (hi - lo)

    hybrid    = lam * mm(pb) + (1 - lam) * mm(pd_)
    top_order = np.argsort(hybrid)[::-1][:k]
    top_idx   = [pool[i] for i in top_order]

    lat = time.perf_counter() - t0
    res = df.iloc[top_idx].copy()
    res["score"] = hybrid[top_order]
    res["rank"]  = range(1, k + 1)
    return res.reset_index(drop=True), lat


# ── RAG generation — brand-aware ──────────────────────────────────────────────
def generate_rag_answer(query, docs):
    context = ""
    for _, row in docs.iterrows():
        brand  = row.get("brand", "")
        ptitle = row.get("product_title", "")
        name   = f"{brand} – {ptitle}".strip(" –") if (brand and brand not in ["", "Unknown Brand"]) else row["title"]
        stars  = row.get("rating", "?")
        context += f"\n[{row['doc_id']}] {name} | ⭐{stars}\n{row['original_text'][:500]}\n"

    # Build ranked brand list for the prompt
    brand_ranking = []
    for _, row in docs.iterrows():
        brand  = row.get("brand", "")
        ptitle = row.get("product_title", "")
        if brand and brand not in ["", "Unknown Brand"]:
            brand_ranking.append(f"Rank #{int(row['rank'])}: {brand} — {ptitle[:50]}")
    brand_context = "\n".join(brand_ranking) if brand_ranking else "No brand data available"

    prompt = f"""You are a knowledgeable e-commerce product advisor.

A customer asked: "{query}"

Top-ranked brands and products retrieved for this query:
{brand_context}

Supporting customer reviews:
{context}

Instructions:
- Open by naming the highest-ranked brand and why it fits the query
- Naturally mention brand names throughout — e.g. "Neutrogena ranks #1 here because..."
- Compare brands if multiple appear
- Cite review IDs like [DOC_ID] as evidence
- End with a clear brand recommendation
- 4-5 sentences maximum

Answer:"""

    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    t0 = time.perf_counter()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip(), time.perf_counter() - t0


# ── UI helpers ────────────────────────────────────────────────────────────────
def review_card(row, color):
    brand  = row.get("brand", "")
    ptitle = row.get("product_title", "")
    stars  = "⭐" * int(float(row.get("rating") or 0))
    preview = row["original_text"][:260]
    if len(row["original_text"]) > 260:
        preview += "…"

    brand_tag = f'<span class="brand-tag">{brand}</span>' if brand and brand not in ["", "Unknown Brand"] else ""
    title_line = ptitle[:60] if ptitle else row["title"][:60]

    st.markdown(f"""
    <div class="review-card" style="border-left-color:{color};">
        {brand_tag}
        <div class="card-title">#{int(row['rank'])} &nbsp; {title_line}</div>
        <div class="card-text">{preview}</div>
        <div class="card-meta">{stars} &nbsp;·&nbsp; score {row['score']:.4f} &nbsp;·&nbsp; {row['doc_id']}</div>
    </div>""", unsafe_allow_html=True)


def brand_visibility_section(hybrid_res):
    """Show which brands appeared and at what rank — embedded GEO insight."""
    brand_ranks = {}
    for _, row in hybrid_res.iterrows():
        brand = row.get("brand", "")
        if brand and brand not in ["", "Unknown Brand"]:
            if brand not in brand_ranks:
                brand_ranks[brand] = int(row["rank"])

    if not brand_ranks:
        return

    st.markdown('<div class="section-label">Brand visibility for this query</div>',
                unsafe_allow_html=True)

    pills = ""
    for brand, rank in sorted(brand_ranks.items(), key=lambda x: x[1]):
        if rank <= 2:
            cls = "brand-top"
            icon = "🟢"
        elif rank <= 4:
            cls = "brand-mid"
            icon = "🟡"
        else:
            cls = "brand-low"
            icon = "🔴"
        pills += f'<span class="brand-pill {cls}">{icon} {brand} — Rank #{rank}</span>'

    st.markdown(f'<div style="margin-bottom:1rem;">{pills}</div>', unsafe_allow_html=True)
    st.caption("🟢 Top 2 · 🟡 Ranks 3–4 · 🔴 Rank 5 — this is your GEO visibility in AI-powered search")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Hero
    st.markdown("""
    <div class="hero">
        <h1>🛍️ ShopSense</h1>
        <p>Ask What You Want, Get What You Need</p>
        <div class="sub">Hybrid RAG · BM25 × FAISS + Claude · Built by Kushala Ralpati Srinivas</div>
    </div>""", unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading enriched reviews and building search indices…"):
        df    = load_data()
        bm25, model, faiss_index = build_indices(df)

    # Query input
    q_col, ex_col = st.columns([3, 1])
    with q_col:
        query = st.text_input("query", label_visibility="collapsed",
            placeholder="🔍  e.g. fragrance free cleanser for sensitive skin")
    with ex_col:
        example = st.selectbox("Examples", ["— try an example —"] + EXAMPLE_QUERIES,
                               label_visibility="collapsed")
    if example != "— try an example —" and not query:
        query = example

    # Empty state
    if not query:
        st.markdown(
            "<p style='color:#94a3b8;text-align:center;margin-top:2rem;'>"
            "👆 Type a product query or pick an example to see the RAG pipeline in action.</p>",
            unsafe_allow_html=True
        )
        return

    # Run retrieval
    with st.spinner("Retrieving…"):
        bm25_res,   bm25_lat = retrieve_bm25(query, bm25, df)
        hybrid_res, hyb_lat  = retrieve_hybrid(query, bm25, model, faiss_index, df)

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("⚡ BM25 Latency",   f"{bm25_lat*1000:.0f} ms")
    m2.metric("🔀 Hybrid Latency", f"{hyb_lat*1000:.0f} ms")
    m3.metric("📄 Reviews searched", f"{len(df):,}")
    m4.metric("🔝 Top results",     TOP_K)

    st.divider()

    # ── Two column layout ──────────────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    # Left — retrieval comparison
    with left:
        st.markdown('<div class="section-label">Retrieval Comparison</div>',
                    unsafe_allow_html=True)
        t1, t2 = st.tabs(["🔵 BM25", "🟣 Hybrid (Best)"])
        with t1:
            st.caption(f"Lexical keyword matching · {bm25_lat*1000:.0f} ms")
            for _, row in bm25_res.iterrows():
                review_card(row, "#2563eb")
        with t2:
            st.caption(f"BM25 + FAISS fused (λ=0.5) · brand-enriched · {hyb_lat*1000:.0f} ms")
            for _, row in hybrid_res.iterrows():
                review_card(row, "#7c3aed")

    # Right — RAG answer + brand visibility
    with right:
        st.markdown('<div class="section-label">RAG Answer (Claude)</div>',
                    unsafe_allow_html=True)
        st.caption("Brand-aware · grounded in Hybrid top-5 reviews")

        try:
            with st.spinner("Generating answer…"):
                answer, gen_lat = generate_rag_answer(query, hybrid_res)

            st.markdown(f'<div class="rag-box">{answer}</div>', unsafe_allow_html=True)
            st.caption(f"⏱️ Generation: {gen_lat:.2f}s")



            # Evidence expanders
            st.markdown('<div class="section-label">Evidence used</div>',
                        unsafe_allow_html=True)
            for _, row in hybrid_res.iterrows():
                brand  = row.get("brand", "")
                ptitle = row.get("product_title", "")
                label  = f"{brand} · {ptitle}".strip(" ·") if brand and brand not in ["","Unknown Brand"] else row["title"]
                with st.expander(f"#{int(row['rank'])}  {label[:65]}"):
                    st.write(row["original_text"][:500])
                    st.caption(f"⭐ {row.get('rating','?')} · {row['doc_id']}")

        except KeyError:
            st.warning("Add `ANTHROPIC_API_KEY` in Streamlit Cloud → Settings → Secrets.")
            st.markdown('<div class="section-label">Evidence (retrieval only)</div>',
                        unsafe_allow_html=True)

            for _, row in hybrid_res.iterrows():
                with st.expander(f"#{int(row['rank'])} {row['title'][:65]}"):
                    st.write(row["original_text"][:400])

    # Footer
    st.markdown("""
    <div class="footer-bar">
        Project · 43,181 reviews · Hybrid nDCG@10=0.7576 · Precision@10=0.8650 · MRR@10=0.9750 &nbsp;|&nbsp;
        <a href="https://github.com/kushalaralpati/Design-and-evaluation-of-RAG-system-for-product-discovery-in-ecommerce">GitHub</a> &nbsp;·&nbsp;
        <a href="https://kushalaportfolio.lovable.app">Portfolio</a>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
