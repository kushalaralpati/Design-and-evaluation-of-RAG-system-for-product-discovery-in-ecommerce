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
    page_title="E-Commerce RAG Demo",
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
.hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 0.3rem 0; }
.hero p  { font-size: 0.95rem; opacity: 0.8; margin: 0.2rem 0; }
.hero .sub { font-size: 0.78rem; opacity: 0.55; margin-top: 0.6rem; }

.metric-row {
    display: flex; gap: 1rem; margin: 1rem 0;
}
.metric-box {
    flex: 1; background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 0.9rem 1.2rem; text-align: center;
}
.metric-box .val { font-size: 1.4rem; font-weight: 700; color: #0f2027; }
.metric-box .lbl { font-size: 0.72rem; color: #64748b; margin-top: 2px; }

.review-card {
    border-radius: 10px; padding: 0.9rem 1rem;
    margin: 0.5rem 0; border-left: 4px solid #ccc;
    background: #fafafa;
}
.review-card .card-title { font-weight: 600; font-size: 0.88rem; margin-bottom: 4px; }
.review-card .card-text  { font-size: 0.82rem; color: #444; line-height: 1.5; }
.review-card .card-meta  { font-size: 0.7rem; color: #999; margin-top: 6px; }

.rag-box {
    background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
    border: 1.5px solid #86efac;
    border-radius: 12px; padding: 1.2rem 1.4rem;
    font-size: 0.9rem; line-height: 1.7; color: #1a2e1a;
}
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

# ── Constants ────────────────────────────────────────────────────────────────
DEMO_SAMPLE = 3000
SEED = 42
TOP_K = 5
CANDIDATE_K = 100

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
    "alcohol free toner for sensitive skin",
    "body lotion for very dry skin",
]

# ── Data & index loading (cached) ────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("prepared_beauty_reviews.csv")
    df = df.dropna(subset=["clean_text"]).reset_index(drop=True)
    df["clean_text"] = df["clean_text"].astype(str)
    df["original_text"] = df["original_text"].fillna(df["clean_text"]).astype(str)
    df["title"] = df["title"].fillna("Review").astype(str)
    df["doc_id"] = df["doc_id"].astype(str)
    return df.sample(n=min(DEMO_SAMPLE, len(df)), random_state=SEED).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def build_indices(_df):
    # BM25 on clean tokenised text
    tokenized = [t.split() for t in _df["clean_text"]]
    bm25 = BM25Okapi(tokenized)

    # Dense index
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(
        _df["clean_text"].tolist(),
        batch_size=128,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return bm25, model, index


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve_bm25(query: str, bm25, df: pd.DataFrame, k: int = TOP_K):
    t0 = time.perf_counter()
    scores = bm25.get_scores(query.lower().split())
    idx = np.argsort(scores)[::-1][:k]
    lat = time.perf_counter() - t0
    res = df.iloc[idx].copy()
    res["score"] = scores[idx]
    res["rank"] = range(1, k + 1)
    return res.reset_index(drop=True), lat


def retrieve_hybrid(query: str, bm25, model, faiss_index, df: pd.DataFrame,
                    k: int = TOP_K, lam: float = 0.5):
    t0 = time.perf_counter()

    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_top = set(np.argsort(bm25_scores)[::-1][:CANDIDATE_K].tolist())

    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    d_scores_raw, d_idx = faiss_index.search(q_emb, CANDIDATE_K)
    dense_top = set(d_idx[0].tolist())
    dense_map = {int(i): float(s) for i, s in zip(d_idx[0], d_scores_raw[0])}

    pool = list(bm25_top | dense_top)
    pb = np.array([bm25_scores[i] for i in pool])
    pd_ = np.array([dense_map.get(i, 0.0) for i in pool])

    def mm(a):
        lo, hi = a.min(), a.max()
        return np.ones_like(a) if hi == lo else (a - lo) / (hi - lo)

    hybrid = lam * mm(pb) + (1 - lam) * mm(pd_)
    top_order = np.argsort(hybrid)[::-1][:k]
    top_idx = [pool[i] for i in top_order]
    top_scores = hybrid[top_order]

    lat = time.perf_counter() - t0
    res = df.iloc[top_idx].copy()
    res["score"] = top_scores
    res["rank"] = range(1, k + 1)
    return res.reset_index(drop=True), lat


# ── RAG generation ────────────────────────────────────────────────────────────
def generate_rag_answer(query: str, docs: pd.DataFrame) -> tuple[str, float]:
    context_blocks = []
    for _, row in docs.iterrows():
        snippet = row["original_text"][:600]
        context_blocks.append(
            f"[{row['doc_id']}] ⭐{row.get('rating','?')} | {row['title']}\n{snippet}"
        )
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful product advisor for an e-commerce beauty store.

Customer query: "{query}"

Customer reviews to use as evidence:
{context}

Instructions:
- Answer using ONLY the reviews above as evidence
- Be specific: name product types and key qualities mentioned
- Cite review IDs like [DOC_ID] when you make a claim  
- If reviews don't directly answer the query, say so honestly
- Keep your answer to 3–5 sentences

Answer:"""

    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    t0 = time.perf_counter()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=350,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip(), time.perf_counter() - t0


# ── Review card ───────────────────────────────────────────────────────────────
def review_card(row, color: str):
    stars = "⭐" * int(float(row.get("rating") or 0))
    preview = row["original_text"][:260]
    if len(row["original_text"]) > 260:
        preview += "…"
    st.markdown(f"""
    <div class="review-card" style="border-left-color:{color};">
        <div class="card-title">#{int(row['rank'])} &nbsp; {row['title'][:60]}</div>
        <div class="card-text">{preview}</div>
        <div class="card-meta">{stars} &nbsp;·&nbsp; score {row['score']:.4f} &nbsp;·&nbsp; {row['doc_id']}</div>
    </div>""", unsafe_allow_html=True)


# ── App ───────────────────────────────────────────────────────────────────────
def main():
    # Hero banner
    st.markdown("""
    <div class="hero">
        <h1>🛍️ E-Commerce Product Discovery</h1>
        <p>Hybrid RAG Pipeline · BM25 × FAISS + Claude Generation</p>
        <div class="sub">MSc Thesis · GISMA Business School Berlin · Kushala Ralpati Srinivas</div>
    </div>""", unsafe_allow_html=True)

    # Load data + build indices
    with st.spinner("Loading 3,000 beauty reviews and building search indices…"):
        df = load_data()
        bm25, model, faiss_index = build_indices(df)

    # Query row
    q_col, ex_col = st.columns([3, 1])
    with q_col:
        query = st.text_input(
            "query", label_visibility="collapsed",
            placeholder="🔍  e.g. fragrance free cleanser for sensitive skin"
        )
    with ex_col:
        example = st.selectbox("Examples", ["— try an example —"] + EXAMPLE_QUERIES,
                               label_visibility="collapsed")

    if example != "— try an example —" and not query:
        query = example

    # Empty state
    if not query:
        st.markdown("""
        <div style="background:#f8fafc; border-radius:12px; padding:1.5rem 2rem; margin-top:1rem;">
            <p style="font-weight:600; margin-bottom:0.8rem;">What this demo shows</p>
            <p>🔵 <b>BM25</b> — fast lexical retrieval (keyword matching)</p>
            <p>🟣 <b>Hybrid</b> — BM25 + FAISS dense embeddings fused with λ=0.5</p>
            <p>🤖 <b>RAG Answer</b> — Claude grounded in the Hybrid top-5 reviews</p>
            <p>⏱️ <b>Latency</b> — live comparison per query</p>
            <p style="color:#94a3b8; font-size:0.8rem; margin-top:1rem;">
                Thesis results on full 43,181 reviews: Hybrid nDCG@10 = 0.7576 · Precision@10 = 0.8650 · MRR@10 = 0.9750
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Run retrieval
    with st.spinner("Retrieving reviews…"):
        bm25_res, bm25_lat = retrieve_bm25(query, bm25, df)
        hybrid_res, hybrid_lat = retrieve_hybrid(query, bm25, model, faiss_index, df)

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("⚡ BM25", f"{bm25_lat * 1000:.0f} ms")
    m2.metric("🔀 Hybrid", f"{hybrid_lat * 1000:.0f} ms")
    m3.metric("📄 Reviews searched", f"{len(df):,}")
    m4.metric("🔝 Results shown", TOP_K)

    st.divider()

    # Two-column layout
    left, right = st.columns([1, 1], gap="large")

    # ── Left: retrieval comparison ──
    with left:
        st.markdown('<div class="section-label">Retrieval Comparison</div>', unsafe_allow_html=True)
        tab_bm25, tab_hybrid = st.tabs(["🔵 BM25", "🟣 Hybrid (Best)"])
        with tab_bm25:
            st.caption(f"Lexical keyword matching · {bm25_lat*1000:.0f} ms")
            for _, row in bm25_res.iterrows():
                review_card(row, "#2563eb")
        with tab_hybrid:
            st.caption(f"BM25 + FAISS fused (λ=0.5) · {hybrid_lat*1000:.0f} ms")
            for _, row in hybrid_res.iterrows():
                review_card(row, "#7c3aed")

    # ── Right: RAG answer ──
    with right:
        st.markdown('<div class="section-label">RAG Answer (Claude)</div>', unsafe_allow_html=True)
        st.caption("Generated from Hybrid top-5 reviews as evidence")

        try:
            with st.spinner("Generating answer…"):
                answer, gen_lat = generate_rag_answer(query, hybrid_res)

            st.markdown(f'<div class="rag-box">{answer}</div>', unsafe_allow_html=True)
            st.caption(f"⏱️ Generation: {gen_lat:.2f}s")

            st.markdown("---")
            st.markdown('<div class="section-label">Evidence used</div>', unsafe_allow_html=True)
            for _, row in hybrid_res.iterrows():
                with st.expander(f"#{int(row['rank'])}  {row['title'][:65]}"):
                    st.write(row["original_text"][:500])
                    st.caption(f"⭐ {row.get('rating','?')} · {row['doc_id']}")

        except KeyError:
            st.warning("Add your `ANTHROPIC_API_KEY` in Streamlit Cloud → App Settings → Secrets to enable generation.")
            st.markdown("**Evidence that would be used:**")
            for _, row in hybrid_res.iterrows():
                with st.expander(f"#{int(row['rank'])}  {row['title'][:65]}"):
                    st.write(row["original_text"][:500])

    # Footer
    st.markdown("""
    <div class="footer-bar">
        Thesis results · 43,181 reviews · Hybrid: nDCG@10=0.7576 · Precision@10=0.8650 · MRR@10=0.9750 &nbsp;|&nbsp;
        <a href="https://github.com/kushalaralpati/Design-and-evaluation-of-RAG-system-for-product-discovery-in-ecommerce">GitHub</a> &nbsp;·&nbsp;
        <a href="https://kushalaportfolio.lovable.app">Portfolio</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
