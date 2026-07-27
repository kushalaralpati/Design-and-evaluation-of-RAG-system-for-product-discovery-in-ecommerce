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

.geo-score-box {
    border-radius: 12px; padding: 1.5rem;
    text-align: center; margin-bottom: 1rem;
}
.geo-score-number { font-size: 3rem; font-weight: 700; }
.geo-found    { background: #f0fdf4; border: 2px solid #86efac; }
.geo-partial  { background: #fffbeb; border: 2px solid #fcd34d; }
.geo-missing  { background: #fef2f2; border: 2px solid #fca5a5; }

.diag-row {
    display: flex; align-items: flex-start; gap: 0.6rem;
    padding: 0.5rem 0; border-bottom: 1px solid #f1f5f9;
    font-size: 0.85rem;
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
DEMO_SAMPLE = 5000
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
]

GEO_QUERIES = [
    "moisturizer for dry skin",
    "fragrance free for sensitive skin",
    "anti aging cream",
    "lightweight sunscreen",
    "gentle cleanser",
    "volumizing shampoo",
    "hydrating serum",
    "waterproof mascara",
    "natural deodorant",
    "hair oil for frizzy hair",
]


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    # Load enriched dataset
    try:
        df = pd.read_csv(
            "enriched_beauty_reviews.csv.gz",
            compression="gzip"
        )
    except Exception:
        df = pd.read_csv("prepared_beauty_reviews.csv")

    df = df.dropna(subset=["clean_text"]).reset_index(drop=True)
    df["clean_text"]     = df["clean_text"].astype(str)
    df["original_text"]  = df.get("original_text", df["clean_text"]).fillna(df["clean_text"]).astype(str)
    df["title"]          = df.get("title", pd.Series(["Review"] * len(df))).fillna("Review").astype(str)
    df["doc_id"]         = df["doc_id"].astype(str)
    df["brand"]          = df.get("brand", pd.Series([""] * len(df))).fillna("").astype(str)
    df["product_title"]  = df.get("product_title", pd.Series([""] * len(df))).fillna("").astype(str)

    # Use enriched_text if available, else fall back to clean_text
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
    embs = model.encode(
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
    t0 = time.perf_counter()
    scores = bm25.get_scores(query.lower().split())
    idx = np.argsort(scores)[::-1][:k]
    lat = time.perf_counter() - t0
    res = df.iloc[idx].copy()
    res["bm25_score"] = scores[idx]
    res["score"] = scores[idx]
    res["rank"] = range(1, k + 1)
    return res.reset_index(drop=True), lat


def retrieve_hybrid(query, bm25, model, faiss_index, df, k=TOP_K, lam=0.5, candidate_k=CANDIDATE_K):
    t0 = time.perf_counter()
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_top = set(np.argsort(bm25_scores)[::-1][:candidate_k].tolist())

    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    d_scores_raw, d_idx = faiss_index.search(q_emb, candidate_k)
    dense_top = set(d_idx[0].tolist())
    dense_map = {int(i): float(s) for i, s in zip(d_idx[0], d_scores_raw[0])}

    pool = list(bm25_top | dense_top)
    pb  = np.array([bm25_scores[i] for i in pool])
    pd_ = np.array([dense_map.get(i, 0.0) for i in pool])

    def mm(a):
        lo, hi = a.min(), a.max()
        return np.ones_like(a) if hi == lo else (a - lo) / (hi - lo)

    hybrid = lam * mm(pb) + (1 - lam) * mm(pd_)
    top_order = np.argsort(hybrid)[::-1][:k]
    top_idx   = [pool[i] for i in top_order]
    top_scores = hybrid[top_order]

    lat = time.perf_counter() - t0
    res = df.iloc[top_idx].copy()
    res["score"]      = top_scores
    res["bm25_score"] = [bm25_scores[i] for i in top_idx]
    res["dense_score"] = [dense_map.get(i, 0.0) for i in top_idx]
    res["rank"] = range(1, k + 1)
    return res.reset_index(drop=True), lat


def retrieve_hybrid_deep(query, bm25, model, faiss_index, df, k=20):
    """Retrieve top-20 for GEO analysis."""
    return retrieve_hybrid(query, bm25, model, faiss_index, df, k=k)


# ── RAG generation ────────────────────────────────────────────────────────────
def generate_rag_answer(query, docs):
    context = ""
    for _, row in docs.iterrows():
        brand = row.get("brand", "")
        ptitle = row.get("product_title", "")
        label = f"{brand} – {ptitle}".strip(" –") if brand or ptitle else row["title"]
        context += f"\n[{row['doc_id']}] {label} ⭐{row.get('rating','?')}\n{row['original_text'][:500]}\n"

    prompt = f"""You are a helpful e-commerce product advisor.

Customer query: "{query}"

Customer reviews (evidence):
{context}

Answer using ONLY these reviews. Be specific, cite review IDs like [DOC_ID], keep to 3-5 sentences."""

    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    t0 = time.perf_counter()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=350,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip(), time.perf_counter() - t0


# ── GEO helpers ───────────────────────────────────────────────────────────────
def geo_score_from_rank(rank):
    mapping = {1: 100, 2: 85, 3: 70, 4: 55, 5: 40}
    if rank <= 5:   return mapping[rank]
    if rank <= 10:  return 20
    if rank <= 20:  return 10
    return 0


def run_geo_analysis(brand_query, df, bm25, model, faiss_index):
    """Run GEO check across multiple queries and return full results."""
    q = brand_query.lower().strip()

    # Find matching products
    brand_mask   = df["brand"].str.lower().str.contains(q, na=False)
    title_mask   = df["product_title"].str.lower().str.contains(q, na=False)
    asin_mask    = df["asin"].str.lower() == q if "asin" in df.columns else pd.Series([False]*len(df))
    doc_mask     = df["doc_id"].str.lower() == q

    matched = df[brand_mask | title_mask | asin_mask | doc_mask]
    return matched


def generate_geo_recommendations(query, brand_name, brand_reviews, top_results, score):
    """Use Claude to generate actionable GEO recommendations."""
    brand_text = "\n".join(brand_reviews["original_text"].head(3).tolist())
    top_text   = "\n".join(top_results["original_text"].head(3).tolist())

    prompt = f"""You are an e-commerce SEO expert specialising in LLM search visibility (GEO - Generative Engine Optimization).

Brand/Product: "{brand_name}"
Customer query: "{query}"
GEO Visibility Score: {score}/100

Brand's current review content (what LLMs see):
{brand_text[:800]}

Top-ranked competitor content for this query:
{top_text[:800]}

Give 3 specific, actionable recommendations to improve this brand's GEO score.
Each recommendation should be 1-2 sentences. Be concrete and specific to the content gap you see.
Format: numbered list 1. 2. 3."""

    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


# ── UI components ─────────────────────────────────────────────────────────────
def review_card(row, color):
    brand = row.get("brand", "")
    ptitle = row.get("product_title", "")
    label = f"{brand} · {ptitle}".strip(" ·") if (brand and brand != "Unknown Brand") else row["title"]
    stars = "⭐" * int(float(row.get("rating") or 0))
    preview = row["original_text"][:250] + "…" if len(row["original_text"]) > 250 else row["original_text"]

    st.markdown(f"""
    <div class="review-card" style="border-left-color:{color};">
        <div class="card-title">#{int(row['rank'])} &nbsp; {label[:65]}</div>
        <div class="card-text">{preview}</div>
        <div class="card-meta">{stars} &nbsp;·&nbsp; score {row['score']:.4f} &nbsp;·&nbsp; {row['doc_id']}</div>
    </div>""", unsafe_allow_html=True)


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="hero">
        <h1>🛍️ E-Commerce Product Discovery</h1>
        <p>Hybrid RAG Pipeline · BM25 × FAISS + Claude Generation · GEO Visibility Check</p>
        <div class="sub">MSc Project · GISMA Business School Berlin · Kushala Ralpati Srinivas</div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Loading enriched reviews and building search indices…"):
        df = load_data()
        bm25, model, faiss_index = build_indices(df)

    # Top-level tabs
    tab_demo, tab_geo = st.tabs(["🔍 Product Discovery Demo", "📊 GEO Visibility Check"])

    # ── TAB 1: DEMO ────────────────────────────────────────────────────────────
    with tab_demo:
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
            st.markdown("""
            <div style="background:#f8fafc;border-radius:12px;padding:1.5rem 2rem;margin-top:1rem;">
                <p style="font-weight:600;margin-bottom:0.8rem;">What this demo shows</p>
                <p>🔵 <b>BM25</b> — fast lexical retrieval (keyword matching)</p>
                <p>🟣 <b>Hybrid</b> — BM25 + FAISS dense embeddings fused with λ=0.5</p>
                <p>🤖 <b>RAG Answer</b> — Claude grounded in Hybrid top-5 reviews + brand metadata</p>
                <p>⏱️ <b>Latency</b> — live comparison per query</p>
                <p style="color:#94a3b8;font-size:0.8rem;margin-top:1rem;">
                Thesis results · 43,181 reviews · Hybrid nDCG@10=0.7576 · Precision@10=0.8650 · MRR@10=0.9750
                </p>
            </div>""", unsafe_allow_html=True)
        else:
            with st.spinner("Retrieving…"):
                bm25_res, bm25_lat   = retrieve_bm25(query, bm25, df)
                hybrid_res, hyb_lat  = retrieve_hybrid(query, bm25, model, faiss_index, df)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("⚡ BM25",           f"{bm25_lat*1000:.0f} ms")
            m2.metric("🔀 Hybrid",         f"{hyb_lat*1000:.0f} ms")
            m3.metric("📄 Reviews",         f"{len(df):,}")
            m4.metric("🔝 Results shown",   TOP_K)
            st.divider()

            left, right = st.columns([1, 1], gap="large")
            with left:
                st.markdown('<div class="section-label">Retrieval Comparison</div>', unsafe_allow_html=True)
                t1, t2 = st.tabs(["🔵 BM25", "🟣 Hybrid (Best)"])
                with t1:
                    st.caption(f"Lexical keyword matching · {bm25_lat*1000:.0f} ms")
                    for _, row in bm25_res.iterrows():
                        review_card(row, "#2563eb")
                with t2:
                    st.caption(f"BM25 + FAISS fused (λ=0.5) · {hyb_lat*1000:.0f} ms")
                    for _, row in hybrid_res.iterrows():
                        review_card(row, "#7c3aed")

            with right:
                st.markdown('<div class="section-label">RAG Answer (Claude)</div>', unsafe_allow_html=True)
                st.caption("Generated from Hybrid top-5 reviews as evidence")
                try:
                    with st.spinner("Generating…"):
                        answer, gen_lat = generate_rag_answer(query, hybrid_res)
                    st.markdown(f'<div class="rag-box">{answer}</div>', unsafe_allow_html=True)
                    st.caption(f"⏱️ Generation: {gen_lat:.2f}s")
                    st.markdown("---")
                    st.markdown('<div class="section-label">Evidence used</div>', unsafe_allow_html=True)
                    for _, row in hybrid_res.iterrows():
                        brand = row.get("brand","")
                        ptitle = row.get("product_title","")
                        label = f"{brand} · {ptitle}".strip(" ·") if brand else row["title"]
                        with st.expander(f"#{int(row['rank'])}  {label[:65]}"):
                            st.write(row["original_text"][:500])
                            st.caption(f"⭐ {row.get('rating','?')} · {row['doc_id']}")
                except KeyError:
                    st.warning("Add `ANTHROPIC_API_KEY` in Streamlit Cloud → Settings → Secrets.")

    # ── TAB 2: GEO CHECK ───────────────────────────────────────────────────────
    with tab_geo:
        st.markdown("""
        <div style="background:#f8fafc;border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1.2rem;">
            <b>🔍 GEO Visibility Check</b> — Is your product showing up when customers ask an AI?<br>
            <span style="font-size:0.85rem;color:#64748b;">
            Enter a brand name or product name below. We'll run 10 real customer queries and 
            score how visible your product is in LLM-powered search results.
            </span>
        </div>""", unsafe_allow_html=True)

        g1, g2 = st.columns([2, 1])
        with g1:
            brand_input = st.text_input("Brand or product name",
                placeholder="e.g. CeraVe, Neutrogena, Maybelline…")
        with g2:
            check_btn = st.button("🔍 Check GEO Visibility", use_container_width=True)

        if not brand_input:
            st.info("Enter a brand or product name above to check its AI search visibility.")

            # Show available brands as a hint
            if "brand" in df.columns:
                known_brands = df[df["brand"].str.len() > 2]["brand"].value_counts().head(15).index.tolist()
                if known_brands:
                    st.markdown("**Brands available in this demo dataset:**")
                    st.write(" · ".join(known_brands))

        elif check_btn or brand_input:
            matched = run_geo_analysis(brand_input, df, bm25, model, faiss_index)

            if len(matched) == 0:
                st.error(f"No products found for **'{brand_input}'** in this dataset. Try a different brand name.")
                if "brand" in df.columns:
                    known = df[df["brand"].str.len() > 2]["brand"].value_counts().head(10).index.tolist()
                    st.write("Try one of these:", " · ".join(known))
            else:
                st.success(f"Found **{len(matched)} reviews** for **'{brand_input}'** across **{matched['asin'].nunique() if 'asin' in matched.columns else '?'} products**")

                # Run GEO scoring across 10 standard queries
                st.markdown("---")
                st.markdown("**Running 10 standard customer queries to score visibility…**")

                results = []
                progress = st.progress(0)

                for i, geo_q in enumerate(GEO_QUERIES):
                    top20, _ = retrieve_hybrid_deep(geo_q, bm25, model, faiss_index, df, k=20)
                    top20_ids = top20["doc_id"].tolist()
                    matched_ids = set(matched["doc_id"].tolist())

                    found_rank = None
                    for rank_idx, doc_id in enumerate(top20_ids):
                        if doc_id in matched_ids:
                            found_rank = rank_idx + 1
                            break

                    score = geo_score_from_rank(found_rank) if found_rank else 0
                    results.append({
                        "query": geo_q,
                        "rank": found_rank,
                        "score": score,
                        "found": found_rank is not None
                    })
                    progress.progress((i + 1) / len(GEO_QUERIES))

                progress.empty()
                results_df = pd.DataFrame(results)

                # ── Overall GEO Score ──
                avg_score = int(results_df["score"].mean())
                found_count = results_df["found"].sum()

                if avg_score >= 60:
                    css_class, emoji, label = "geo-found",    "✅", "Good Visibility"
                elif avg_score >= 25:
                    css_class, emoji, label = "geo-partial",  "⚠️", "Partial Visibility"
                else:
                    css_class, emoji, label = "geo-missing",  "❌", "Low Visibility"

                col_score, col_detail = st.columns([1, 2])

                with col_score:
                    st.markdown(f"""
                    <div class="geo-score-box {css_class}">
                        <div class="geo-score-number">{avg_score}</div>
                        <div style="font-size:0.9rem;font-weight:600;">GEO Score / 100</div>
                        <div style="font-size:1.2rem;margin-top:0.4rem;">{emoji} {label}</div>
                        <div style="font-size:0.8rem;color:#64748b;margin-top:0.4rem;">
                            Found in {found_count}/10 customer queries
                        </div>
                    </div>""", unsafe_allow_html=True)

                    # Score guide
                    st.markdown("""
                    <div style="font-size:0.75rem;color:#94a3b8;margin-top:0.5rem;">
                    <b>Score guide:</b><br>
                    🟢 70–100 · Strong visibility<br>
                    🟡 30–69 · Partial — needs work<br>
                    🔴 0–29 · Invisible to AI search
                    </div>""", unsafe_allow_html=True)

                with col_detail:
                    st.markdown("**Query-by-query breakdown:**")
                    for _, r in results_df.iterrows():
                        if r["found"]:
                            rank_label = f"Rank #{int(r['rank'])}"
                            icon = "🟢" if r["rank"] <= 5 else "🟡"
                        else:
                            rank_label = "Not found"
                            icon = "🔴"
                        st.markdown(
                            f"{icon} **{r['query']}** — {rank_label} · Score: {int(r['score'])}/100"
                        )

                # ── Diagnostics ──
                st.markdown("---")
                st.markdown("**🔬 Why this score?**")

                d1, d2 = st.columns(2)
                with d1:
                    # Content length analysis
                    brand_avg_words = matched["original_text"].str.split().str.len().mean()

                    # Get top ranked docs for comparison
                    sample_query = GEO_QUERIES[0]
                    top5, _ = retrieve_hybrid(sample_query, bm25, model, faiss_index, df)
                    top5_avg_words = top5["original_text"].str.split().str.len().mean()

                    st.markdown(f"""
                    📝 **Review length**
                    - Your reviews: **{brand_avg_words:.0f} words avg**
                    - Top-ranked competitors: **{top5_avg_words:.0f} words avg**
                    - {'✅ Good length' if brand_avg_words >= top5_avg_words * 0.8 else '⚠️ Your reviews are shorter — add more detail'}
                    """)

                with d2:
                    # Keyword coverage
                    all_brand_text = " ".join(matched["enriched_text"].tolist()).lower()
                    covered = sum(1 for q in GEO_QUERIES
                                  if any(w in all_brand_text for w in q.split()))
                    st.markdown(f"""
                    🔑 **Query keyword coverage**
                    - Your content covers **{covered}/10** standard queries
                    - {'✅ Strong keyword presence' if covered >= 7 else '⚠️ Missing keywords for ' + str(10-covered) + ' query types'}

                    🏷️ **Brand mentions in reviews**
                    - Brand name appears in enriched text: {'✅ Yes' if brand_input.lower() in all_brand_text else '❌ No — brand not mentioned in reviews'}
                    """)

                # ── AI Recommendations ──
                st.markdown("---")
                st.markdown("**💡 How to improve your GEO score**")

                try:
                    with st.spinner("Generating recommendations…"):
                        worst_query = results_df.sort_values("score").iloc[0]["query"]
                        top_results, _ = retrieve_hybrid(worst_query, bm25, model, faiss_index, df)
                        recs = generate_geo_recommendations(
                            worst_query, brand_input, matched, top_results, avg_score
                        )
                    st.markdown(f'<div class="rag-box">{recs}</div>', unsafe_allow_html=True)
                    st.caption(f"Recommendations based on weakest query: *'{worst_query}'*")
                except KeyError:
                    st.warning("Add `ANTHROPIC_API_KEY` in Streamlit Cloud → Settings → Secrets for AI recommendations.")

                # ── Show brand's reviews ──
                st.markdown("---")
                with st.expander(f"📋 See all {len(matched)} reviews found for '{brand_input}'"):
                    for _, row in matched.head(10).iterrows():
                        st.markdown(f"**{row.get('product_title','')[:80]}** · ⭐{row.get('rating','?')}")
                        st.write(row["original_text"][:300])
                        st.divider()

    # Footer
    st.markdown("""
    <div class="footer-bar">
        Project results · 43,181 reviews · Hybrid: nDCG@10=0.7576 · Precision@10=0.8650 · MRR@10=0.9750 &nbsp;|&nbsp;
        <a href="https://github.com/kushalaralpati/Design-and-evaluation-of-RAG-system-for-product-discovery-in-ecommerce">GitHub</a> &nbsp;·&nbsp;
        <a href="https://kushalaportfolio.lovable.app">Portfolio</a>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
