import re
import time
import numpy as np
import pandas as pd
import streamlit as st
import faiss
import anthropic
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopSense — AI Product Discovery",
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

.pipeline-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.pipeline-stage {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.82rem;
    padding: 0.3rem 0;
    color: #475569;
}
.stage-done  { color: #16a34a; font-weight: 600; }
.stage-time  { color: #94a3b8; font-size: 0.72rem; margin-left: auto; }

.expanded-query {
    background: #ede9fe;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    color: #5b21b6;
    margin-bottom: 1rem;
}

.review-card {
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid #ccc;
    background: #fafafa;
}
.brand-tag {
    display: inline-block;
    background: #ede9fe; color: #5b21b6;
    font-size: 0.7rem; font-weight: 600;
    padding: 2px 8px; border-radius: 99px;
    margin-bottom: 5px;
}
.score-row {
    display: flex; gap: 0.8rem;
    font-size: 0.7rem; color: #94a3b8;
    margin-top: 4px;
}
.score-highlight { color: #7c3aed; font-weight: 600; }

.rag-box {
    background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
    border: 1.5px solid #86efac;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    font-size: 0.92rem;
    line-height: 1.8;
    color: #1a2e1a;
    margin-bottom: 1rem;
}

.compressed-badge {
    display: inline-block;
    background: #fef3c7; color: #92400e;
    font-size: 0.65rem; font-weight: 600;
    padding: 1px 6px; border-radius: 99px;
    margin-left: 6px;
}

.section-label {
    font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8; margin-bottom: 0.5rem;
}
.footer-bar {
    text-align: center; color: #94a3b8;
    font-size: 0.75rem;
    padding: 1.2rem 0 0.5rem 0;
    border-top: 1px solid #e2e8f0;
    margin-top: 1.5rem;
}
.footer-bar a { color: #7c3aed; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DEMO_SAMPLE  = 5000
SEED         = 42
RETRIEVAL_K  = 50   # candidates for reranker
FINAL_K      = 5    # final results after reranking
CANDIDATE_K  = 100  # hybrid candidate pool

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

    if "enriched_text" in df.columns:
        df["enriched_text"] = df["enriched_text"].fillna(df["clean_text"]).astype(str)
    else:
        df["enriched_text"] = (
            df["brand"] + " " + df["product_title"] + " " + df["clean_text"]
        ).str.strip()

    return df.sample(n=min(DEMO_SAMPLE, len(df)), random_state=SEED).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def build_indices(_df):
    # BM25
    tokenized = [t.split() for t in _df["enriched_text"]]
    bm25 = BM25Okapi(tokenized)

    # Bi-encoder for FAISS
    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    embs = bi_encoder.encode(
        _df["enriched_text"].tolist(),
        batch_size=128,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    # Cross-encoder for reranking
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return bm25, bi_encoder, index, cross_encoder


# ── Stage 1: Query Expansion ──────────────────────────────────────────────────
def expand_query(query: str, client) -> tuple[str, list[str]]:
    """Expand query with related terms using Claude Haiku."""
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            messages=[{"role": "user", "content":
                f"""You are a beauty product search expert.
Expand this search query with 4-5 related beauty/skincare terms that would help find relevant products.
Return ONLY a comma-separated list of terms. No explanations.

Query: {query}

Related terms:"""}]
        )
        expansion_text = msg.content[0].text.strip()
        extra_terms = [t.strip() for t in expansion_text.split(",") if t.strip()][:5]
        expanded = f"{query} {' '.join(extra_terms)}"
        return expanded, extra_terms
    except Exception:
        return query, []


# ── Stage 2: Hybrid Retrieval ─────────────────────────────────────────────────
def retrieve_hybrid(query: str, bm25, bi_encoder, faiss_index, df,
                    k: int = RETRIEVAL_K, lam: float = 0.5):
    t0 = time.perf_counter()
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_top    = set(np.argsort(bm25_scores)[::-1][:CANDIDATE_K].tolist())

    q_emb              = bi_encoder.encode([query], normalize_embeddings=True).astype("float32")
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
    res["hybrid_score"] = hybrid[top_order]
    res["bm25_score"]   = [bm25_scores[i] for i in top_idx]
    return res.reset_index(drop=True), lat


# ── Stage 3: Cross-Encoder Reranking ─────────────────────────────────────────
def rerank(query: str, candidates: pd.DataFrame,
           cross_encoder, k: int = FINAL_K) -> tuple[pd.DataFrame, float]:
    t0 = time.perf_counter()
    pairs  = [(query, str(row["original_text"])[:512])
              for _, row in candidates.iterrows()]
    scores = cross_encoder.predict(pairs)
    top_idx = scores.argsort()[::-1][:k]

    reranked = candidates.iloc[top_idx].copy()
    reranked["rerank_score"] = scores[top_idx]
    reranked["rank"]         = range(1, k + 1)
    lat = time.perf_counter() - t0
    return reranked.reset_index(drop=True), lat


# ── Stage 4: Contextual Compression ──────────────────────────────────────────
def compress_context(query: str, review_text: str,
                     bi_encoder) -> str:
    """Extract the 3 most query-relevant sentences from a review."""
    text    = re.sub(r"<br\s*/?>", " ", str(review_text))
    sents   = [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) > 15]
    if len(sents) <= 2:
        return text[:300]

    q_emb    = bi_encoder.encode([query], normalize_embeddings=True)
    s_embs   = bi_encoder.encode(sents, normalize_embeddings=True)
    scores   = (q_emb @ s_embs.T)[0]
    top_idx  = sorted(scores.argsort()[::-1][:3].tolist())
    return ". ".join([sents[i] for i in top_idx])


def compress_all(query: str, docs: pd.DataFrame,
                 bi_encoder) -> pd.DataFrame:
    t0 = time.perf_counter()
    docs = docs.copy()
    docs["compressed_text"] = docs["original_text"].apply(
        lambda t: compress_context(query, t, bi_encoder)
    )
    lat = time.perf_counter() - t0
    return docs, lat


# ── Stage 5: RAG Generation ───────────────────────────────────────────────────
def generate_answer(query: str, docs: pd.DataFrame, client) -> tuple[str, float]:
    brand_ranking = []
    context       = ""
    for _, row in docs.iterrows():
        brand  = row.get("brand", "")
        ptitle = row.get("product_title", "")
        name   = f"{brand} – {ptitle}".strip(" –") \
                 if brand and brand not in ["", "Unknown Brand"] else row["title"]

        if brand and brand not in ["", "Unknown Brand"]:
            brand_ranking.append(f"Rank #{int(row['rank'])}: {brand} — {ptitle[:50]}")

        compressed = str(row.get("compressed_text", row["original_text"]))[:400]
        context   += f"\n[{row['doc_id']}] {name} ⭐{row.get('rating','?')}\n{compressed}\n"

    brand_ctx = "\n".join(brand_ranking) if brand_ranking else "No brand data"

    prompt = f"""You are a knowledgeable e-commerce product advisor.

Customer query: "{query}"

Top-ranked brands retrieved:
{brand_ctx}

Compressed customer review evidence:
{context}

Instructions:
- Open by naming the highest-ranked brand and why it fits
- Weave brand names naturally — "Neutrogena ranks #1 because..."
- Compare brands if multiple appear
- Cite review IDs like [DOC_ID]
- End with a clear brand recommendation
- 4-5 sentences maximum

Answer:"""

    t0  = time.perf_counter()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip(), time.perf_counter() - t0


# ── UI helpers ────────────────────────────────────────────────────────────────
def render_pipeline_stages(timings: dict, expanded_terms: list):
    total = sum(timings.values())
    stages = [
        ("🔍", "Query Expansion",        "expansion",  expanded_terms),
        ("📚", "Hybrid Retrieval (×50)", "retrieval",  None),
        ("🎯", "Cross-Encoder Reranking","reranking",  None),
        ("✂️", "Contextual Compression", "compression",None),
        ("🤖", "RAG Generation",         "generation", None),
    ]
    with st.expander(f"⚡ Full pipeline · {total:.2f}s total", expanded=False):
        for icon, name, key, extra in stages:
            t = timings.get(key, 0)
            bar_pct = int((t / total) * 100) if total > 0 else 0
            st.markdown(
                f"{icon} **{name}** "
                f"<span style='color:#94a3b8;font-size:0.75rem;'>"
                f"{t*1000:.0f}ms ({bar_pct}%)</span>",
                unsafe_allow_html=True
            )
            st.progress(bar_pct)
            if extra:
                st.caption("Expanded terms: " + " · ".join(extra))


def render_review_card(row, color: str):
    brand  = row.get("brand", "")
    ptitle = row.get("product_title", "")
    rating = row.get("rating") or 0
    stars  = "⭐" * int(float(rating))

    preview = re.sub(r"<br\s*/?>", " ", str(row["original_text"]))[:260]
    if len(str(row["original_text"])) > 260:
        preview += "…"
    title_line = str(ptitle)[:60] if ptitle else str(row["title"])[:60]

    st.markdown(
        f'<div style="border-left:4px solid {color};'
        f'padding:0.2rem 0 0.2rem 0.8rem;margin:0.5rem 0;">',
        unsafe_allow_html=True
    )
    if brand and brand not in ["", "Unknown Brand"]:
        st.markdown(
            f'<span class="brand-tag">{brand}</span>',
            unsafe_allow_html=True
        )
    st.markdown(f"**#{int(row['rank'])}  {title_line}**")
    st.write(preview)

    rerank = row.get("rerank_score")
    hybrid = row.get("hybrid_score", row.get("score", 0))
    if rerank is not None:
        st.caption(
            f"{stars} · "
            f"🎯 rerank {rerank:.3f} · "
            f"hybrid {hybrid:.4f} · "
            f"{row['doc_id']}"
        )
    else:
        st.caption(f"{stars} · score {hybrid:.4f} · {row['doc_id']}")

    st.markdown("</div>", unsafe_allow_html=True)


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="hero">
        <h1>🛍️ ShopSense</h1>
        <p>Ask What You Want, Get What You Need</p>
        <div class="sub">
            Query Expansion · Hybrid RAG · Cross-Encoder Reranking ·
            Contextual Compression · Claude Generation ·
            Built by Kushala Ralpati Srinivas
        </div>
    </div>""", unsafe_allow_html=True)

    # Load
    with st.spinner("Loading enriched reviews and building indices…"):
        df = load_data()
        bm25, bi_encoder, faiss_index, cross_encoder = build_indices(df)

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

    if not query:
        st.markdown(
            "<p style='color:#94a3b8;text-align:center;margin-top:2rem;'>"
            "👆 Type a product query or pick an example to see the full RAG pipeline.</p>",
            unsafe_allow_html=True
        )
        st.markdown("""
        <div style="background:#f8fafc;border-radius:12px;padding:1.5rem 2rem;margin-top:1rem;">
            <p style="font-weight:600;margin-bottom:0.8rem;">🚀 Pipeline (v2 — 4-stage)</p>
            <p>1️⃣ <b>Query Expansion</b> — Claude adds related terms to your query for better recall</p>
            <p>2️⃣ <b>Hybrid Retrieval</b> — BM25 + FAISS retrieves top-50 candidates</p>
            <p>3️⃣ <b>Cross-Encoder Reranking</b> — Precision reranker selects best top-5</p>
            <p>4️⃣ <b>Contextual Compression</b> — Only the most relevant sentences go to Claude</p>
            <p>5️⃣ <b>Brand-Aware Generation</b> — Claude answers citing brands and review IDs</p>
        </div>""", unsafe_allow_html=True)
        return

    # ── Run full pipeline ──────────────────────────────────────────────────────
    try:
        client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    except KeyError:
        client = None
        st.warning("Add `ANTHROPIC_API_KEY` in Streamlit Cloud → Settings → Secrets.")

    timings       = {}
    expanded_terms = []

    with st.spinner("Running pipeline…"):

        # Stage 1: Query expansion
        if client:
            t0 = time.perf_counter()
            expanded_query, expanded_terms = expand_query(query, client)
            timings["expansion"] = time.perf_counter() - t0
        else:
            expanded_query = query
            timings["expansion"] = 0.0

        # Stage 2: Hybrid retrieval (top-50)
        candidates, timings["retrieval"] = retrieve_hybrid(
            expanded_query, bm25, bi_encoder, faiss_index, df, k=RETRIEVAL_K
        )

        # Stage 3: Cross-encoder reranking (top-5)
        reranked, timings["reranking"] = rerank(
            query, candidates, cross_encoder, k=FINAL_K
        )

        # Stage 4: Contextual compression
        reranked, timings["compression"] = compress_all(query, reranked, bi_encoder)

        # Stage 5: RAG generation
        answer, gen_lat = "", 0.0
        if client:
            t0 = time.perf_counter()
            answer, timings["generation"] = generate_answer(query, reranked, client)
        else:
            timings["generation"] = 0.0

    # ── Metrics row ────────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("📚 Candidates",     f"{len(candidates)}")
    m2.metric("🎯 After rerank",   FINAL_K)
    m3.metric("🔍 Reviews",        f"{len(df):,}")
    m4.metric("⚡ Retrieval",      f"{timings['retrieval']*1000:.0f} ms")
    m5.metric("🎯 Reranking",      f"{timings['reranking']*1000:.0f} ms")

    # Pipeline stages expander
    render_pipeline_stages(timings, expanded_terms)

    # Expanded query display
    if expanded_terms:
        st.markdown(
            f'<div class="expanded-query">'
            f'🔍 <b>Query expanded with:</b> {" · ".join(expanded_terms)}'
            f'</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # ── Two column layout ──────────────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    # Left: retrieval results
    with left:
        st.markdown('<div class="section-label">Reranked Results</div>',
                    unsafe_allow_html=True)
        st.caption(
            f"Top-{RETRIEVAL_K} hybrid → cross-encoder reranked → top-{FINAL_K} · "
            f"{timings['reranking']*1000:.0f}ms reranking"
        )
        for _, row in reranked.iterrows():
            render_review_card(row, "#7c3aed")

    # Right: RAG answer
    with right:
        st.markdown('<div class="section-label">RAG Answer (Claude)</div>',
                    unsafe_allow_html=True)
        st.caption("Brand-aware · compressed evidence · cross-encoder grounded")

        if answer:
            st.markdown(f'<div class="rag-box">{answer}</div>',
                        unsafe_allow_html=True)
            st.caption(f"⏱️ Generation: {timings.get('generation',0):.2f}s")
        elif not client:
            st.info("Add API key to enable generation.")

        # Compressed evidence
        st.markdown("---")
        st.markdown('<div class="section-label">Compressed Evidence Used</div>',
                    unsafe_allow_html=True)
        st.caption("Only the most query-relevant sentences sent to Claude")

        for _, row in reranked.iterrows():
            brand  = row.get("brand", "")
            ptitle = row.get("product_title", "")
            label  = f"{brand} · {ptitle}".strip(" ·") \
                     if brand and brand not in ["","Unknown Brand"] else row["title"]

            with st.expander(
                f"#{int(row['rank'])}  {label[:55]} "
                f"🎯 {row.get('rerank_score',0):.3f}"
            ):
                compressed = str(row.get("compressed_text",""))
                original   = str(row["original_text"])

                col_c, col_o = st.columns(2)
                with col_c:
                    st.markdown("**🟡 Compressed (sent to Claude)**")
                    st.write(compressed)
                with col_o:
                    st.markdown("**📄 Full review**")
                    st.write(original[:400] + ("…" if len(original) > 400 else ""))

                reduction = (1 - len(compressed)/max(len(original),1)) * 100
                st.caption(
                    f"⭐ {row.get('rating','?')} · "
                    f"rerank score {row.get('rerank_score',0):.4f} · "
                    f"context reduced {reduction:.0f}% · "
                    f"{row['doc_id']}"
                )

    # Footer
    st.markdown("""
    <div class="footer-bar">
        ShopSense v2 · Query Expansion · Hybrid BM25+FAISS · Cross-Encoder Reranking ·
        Contextual Compression · Claude Generation &nbsp;|&nbsp;
        <a href="https://github.com/kushalaralpati/Design-and-evaluation-of-RAG-system-for-product-discovery-in-ecommerce">GitHub</a>
        &nbsp;·&nbsp;
        <a href="https://kushalaportfolio.lovable.app">Portfolio</a>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
