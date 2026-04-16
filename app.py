"""
TaosLab Drug Repurposing Platform
==================================
LLM-guided drug repurposing across diseases.
Currently featuring: Anthracycline-Induced Cardiotoxicity (AIC).

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="PriorRx: AI-Prioritized Drug Repurposing Candidates Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0D1B2A 0%, #1B3A5C 40%, #2A7F8E 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.9rem; letter-spacing: 0.5px; }
    .main-header .subtitle { color: #8EC8D8; margin: 0.2rem 0 0 0; font-size: 0.95rem; }
    .main-header .disease-tag {
        display: inline-block; background: rgba(255,255,255,0.15);
        padding: 3px 12px; border-radius: 20px; font-size: 0.8rem;
        color: #B0D4E8; margin-top: 0.5rem;
    }
    .badge-positive { background: #E8F5E9; color: #2E7D32; padding: 3px 10px; border-radius: 4px; font-weight: 600; }
    .badge-adverse  { background: #FFEBEE; color: #C62828; padding: 3px 10px; border-radius: 4px; font-weight: 600; }
    .badge-neutral  { background: #FFF3E0; color: #E65100; padding: 3px 10px; border-radius: 4px; font-weight: 600; }
    .badge-nosign   { background: #ECEFF1; color: #546E7A; padding: 3px 10px; border-radius: 4px; font-weight: 600; }
    .validated-badge {
        background: #FFD600; color: #333; padding: 2px 8px; border-radius: 4px;
        font-weight: 700; font-size: 0.75rem; vertical-align: middle;
    }
    .pmid-link { color: #1565C0; text-decoration: none; font-weight: 600; }
    .pmid-link:hover { text-decoration: underline; }
    .abstract-card {
        border-radius: 8px; padding: 0.8rem 1rem; margin: 0.4rem 0;
        border-left: 4px solid; font-size: 0.9rem;
    }
    .abstract-positive { border-left-color: #4CAF50; background: #F1F8E9; }
    .abstract-negative { border-left-color: #EF5350; background: #FFF8F7; }
    .abstract-neutral  { border-left-color: #BDBDBD; background: #FAFAFA; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Load data ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DRUG_CSV = os.path.join(BASE_DIR, "clinicaltrials_cardiomyopathy_drugs_results_aggregated.csv")
ABSTRACT_CSV = os.path.join(BASE_DIR, "Copy_of_realtime_abstract_analysis__1_.csv")

@st.cache_data
def load_drugs():
    df = pd.read_csv(DRUG_CSV)
    df = df.drop_duplicates(subset=["drug_name", "Model"], keep="first")
    df["rank"] = df["rank"].fillna(0).astype(int)
    df["has_trial"] = df["has_clinical_trial"] == "Yes"
    model_counts = df.groupby("drug_name")["Model"].nunique().rename("n_models")
    df = df.merge(model_counts, on="drug_name", how="left")
    df["validated_in_paper"] = df["drug_name"].isin(
        ["Cysteine", "Dasatinib", "Tranilast", "Tretinoin"]
    )
    return df

@st.cache_data
def load_abstracts():
    if not os.path.exists(ABSTRACT_CSV):
        return pd.DataFrame()
    ab = pd.read_csv(ABSTRACT_CSV)
    # Standardize column names
    ab = ab.rename(columns={
        "Drug_ID": "drug_id", "Drug_name": "drug_name",
        "Disease_ID": "disease_id", "Disease_name": "disease_name",
        "PubMed_ID": "pubmed_id", "Title": "title",
        "Result": "result", "Explanation": "explanation",
        "Raw_Output": "raw_output", "Model": "model"
    })
    return ab

df = load_drugs()
ab_df = load_abstracts()

VALIDATED_DRUGS = {"Cysteine", "Dasatinib", "Tranilast", "Tretinoin"}

def badge(label):
    if label == "Potentially Therapeutic":
        return '<span class="badge-positive">✓ Potentially Therapeutic</span>'
    elif label == "Potentially Adverse Effect":
        return '<span class="badge-adverse">✗ Potentially Adverse</span>'
    elif label == "Neutral Relation":
        return '<span class="badge-neutral">○ Neutral Relation</span>'
    elif label == "No Positive Sign":
        return '<span class="badge-nosign">— No Positive Sign</span>'
    return f'<span class="badge-neutral">{label}</span>'


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 Filters")

    models = st.multiselect("Graph model source",
                            sorted(df["Model"].unique()),
                            default=sorted(df["Model"].unique()))

    st.markdown("---")
    st.markdown("**LLM classification**")
    c1_opts = sorted(df["Criterion_1"].unique())
    c1_filter = st.multiselect("Criterion 1 (conservative)", c1_opts, default=c1_opts)
    c2_opts = sorted(df["Criterion_2"].unique())
    c2_filter = st.multiselect("Criterion 2 (sensitive)", c2_opts, default=c2_opts)

    st.markdown("---")
    st.markdown("**Evidence filters**")
    min_pos = st.slider("Min positive abstracts", 0, int(df["Positive"].max()), 0)
    min_analyzed = st.slider("Min analyzed abstracts", 0, 200, 0)

    st.markdown("---")
    st.markdown("**Clinical trial status**")
    trial_opt = st.radio("Show drugs with trials",
                         ["All", "Has trials", "No trials (novel)"], index=0)

    st.markdown("---")
    search = st.text_input("🔎 Search drug name", "")


# ═══════════════════════════════════════════════════════════════════
# APPLY FILTERS
# ═══════════════════════════════════════════════════════════════════
filt = df[
    (df["Model"].isin(models)) &
    (df["Criterion_1"].isin(c1_filter)) &
    (df["Criterion_2"].isin(c2_filter)) &
    (df["Positive"] >= min_pos) &
    (df["Analyzed"] >= min_analyzed)
]
if trial_opt == "Has trials":
    filt = filt[filt["has_trial"]]
elif trial_opt == "No trials (novel)":
    filt = filt[~filt["has_trial"]]
if search:
    filt = filt[filt["drug_name"].str.contains(search, case=False, na=False)]


# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🧬 PriorRx: AI-Prioritized Drug Repurposing Candidates Platform</h1>
    <p class="subtitle">AI-prioritized drug repurposing through graph learning and LLM-guided evidence synthesis</p>
    <span class="disease-tag">📌 Current module: Anthracycline-Induced Cardiotoxicity (AIC)</span>
</div>
""", unsafe_allow_html=True)

# Metrics
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Total entries", len(df))
m2.metric("Showing", len(filt))
m3.metric("Unique drugs", df["drug_name"].nunique())
m4.metric("Therapeutic (C1)", filt[filt["Criterion_1"] == "Potentially Therapeutic"]["drug_name"].nunique())
m5.metric("Therapeutic (C2)", filt[filt["Criterion_2"] == "Potentially Therapeutic"]["drug_name"].nunique())
novel_count = filt[(filt["Criterion_2"] == "Potentially Therapeutic") & (~filt["has_trial"])]["drug_name"].nunique()
m6.metric("Novel candidates", novel_count)

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════
tab_table, tab_viz, tab_detail, tab_novel = st.tabs(
    ["📋 Drug Table", "📊 Visualizations", "🔬 Drug Detail", "🌟 Novel Candidates"]
)

# ─── TAB 1: DRUG TABLE ───────────────────────────────────────────
with tab_table:
    st.markdown("### All candidate drugs")
    st.info("💡 **Tip:** Switch to the **🔬 Drug Detail** tab and select any drug to view individual abstract classifications, PubMed links, and LLM explanations.")

    display = filt[[
        "drug_name", "Model", "rank", "Criterion_1", "Criterion_2",
        "Analyzed", "Positive", "Neutral", "Negative",
        "Rate_Positive", "Rate_Negative",
        "has_clinical_trial", "clinical_trial_count",
    ]].copy()

    display.columns = [
        "Drug", "Model", "Rank", "Criterion 1", "Criterion 2",
        "Analyzed", "Pos", "Neu", "Neg",
        "Rp", "Rn",
        "Trial?", "# Trials",
    ]

    def color_c1(val):
        if val == "Potentially Therapeutic":
            return "background-color: #E8F5E9; color: #2E7D32; font-weight: 600"
        elif val == "Potentially Adverse Effect":
            return "background-color: #FFEBEE; color: #C62828; font-weight: 600"
        return "background-color: #FFF3E0; color: #E65100"

    def color_c2(val):
        if val == "Potentially Therapeutic":
            return "background-color: #E8F5E9; color: #2E7D32; font-weight: 600"
        return "background-color: #ECEFF1; color: #546E7A"

    styled = display.style\
        .map(color_c1, subset=["Criterion 1"])\
        .map(color_c2, subset=["Criterion 2"])\
        .format({"Rp": "{:.3f}", "Rn": "{:.3f}"})

    st.dataframe(styled, use_container_width=True, height=600)

    csv_out = filt.to_csv(index=False)
    st.download_button("📥 Download filtered results (CSV)", csv_out,
                       "aic_candidates_filtered.csv", "text/csv")


# ─── TAB 2: VISUALIZATIONS ──────────────────────────────────────
with tab_viz:
    v1, v2 = st.columns(2)

    with v1:
        st.markdown("#### Criterion 1 by model")
        c1c = filt.groupby(["Model", "Criterion_1"]).size().reset_index(name="count")
        cmap1 = {"Potentially Therapeutic": "#4CAF50",
                 "Potentially Adverse Effect": "#EF5350",
                 "Neutral Relation": "#FF9800"}
        fig1 = px.bar(c1c, x="Model", y="count", color="Criterion_1",
                      color_discrete_map=cmap1, barmode="stack",
                      labels={"count": "Drugs", "Criterion_1": ""})
        fig1.update_layout(height=340, margin=dict(t=10, b=30),
                           legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig1, use_container_width=True)

    with v2:
        st.markdown("#### Criterion 2 by model")
        c2c = filt.groupby(["Model", "Criterion_2"]).size().reset_index(name="count")
        cmap2 = {"Potentially Therapeutic": "#4CAF50",
                 "No Positive Sign": "#9E9E9E"}
        fig2 = px.bar(c2c, x="Model", y="count", color="Criterion_2",
                      color_discrete_map=cmap2, barmode="stack",
                      labels={"count": "Drugs", "Criterion_2": ""})
        fig2.update_layout(height=340, margin=dict(t=10, b=30),
                           legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter: rank vs positive rate
    st.markdown("#### Positive rate (Rp) by model rank")
    scatter = filt[filt["rank"] > 0].copy()
    if not scatter.empty:
        scatter["label"] = scatter.apply(
            lambda r: r["drug_name"] if r["validated_in_paper"] else "", axis=1)
        fig4 = px.scatter(scatter, x="rank", y="Rate_Positive",
                          color="Model", size="Analyzed",
                          hover_name="drug_name", text="label",
                          labels={"rank": "Model rank (lower = higher priority)",
                                  "Rate_Positive": "Positive rate (Rp)",
                                  "Analyzed": "Abstracts"},
                          color_discrete_sequence=["#1B3A5C", "#2A7F8E", "#E8734A"])
        fig4.update_traces(textposition="top center", textfont_size=11)
        fig4.update_layout(height=420, margin=dict(t=10, b=30))
        st.plotly_chart(fig4, use_container_width=True)

    # Top drugs by positive count
    st.markdown("#### Top drugs by positive abstract count")
    top_pos = filt[filt["Positive"] > 0].sort_values("Positive", ascending=True).tail(15)
    if not top_pos.empty:
        fig5 = px.bar(top_pos, y="drug_name", x="Positive", orientation="h",
                      color="Criterion_2",
                      color_discrete_map={"Potentially Therapeutic": "#4CAF50",
                                          "No Positive Sign": "#BDBDBD"},
                      hover_data=["Model", "Analyzed", "Rate_Positive"],
                      labels={"drug_name": "", "Positive": "Positive abstracts"})
        fig5.update_layout(height=400, margin=dict(t=10, b=30, l=140),
                           showlegend=True, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig5, use_container_width=True)


# ─── TAB 3: DRUG DETAIL ─────────────────────────────────────────
with tab_detail:
    st.markdown("### 🔬 Drug detail view")

    drug_names = sorted(filt["drug_name"].unique())
    if not drug_names:
        st.warning("No drugs match current filters.")
    else:
        default_idx = 0
        for i, d in enumerate(drug_names):
            if d in VALIDATED_DRUGS:
                default_idx = i
                break
        selected = st.selectbox("Select a drug", drug_names, index=default_idx)

        rows = filt[filt["drug_name"] == selected]
        row = rows.iloc[0]

        # ── Header ──
        hc1, hc2, hc3 = st.columns([2, 1, 1])
        with hc1:
            v_extra = (' <span class="validated-badge">✓ VALIDATED IN VIVO</span>'
                       if row["validated_in_paper"] else "")
            st.markdown(f"## {selected}{v_extra}", unsafe_allow_html=True)
            if len(rows) > 1:
                st.markdown(f"Appears in **{len(rows)} models**: {', '.join(rows['Model'].tolist())}")
        with hc2:
            st.markdown(f"**Model:** {row['Model']}")
            st.markdown(f"**Rank:** #{row['rank']}" if row['rank'] > 0 else "**Rank:** N/A")
        with hc3:
            st.markdown(f"**Criterion 1:** {badge(row['Criterion_1'])}", unsafe_allow_html=True)
            st.markdown(f"**Criterion 2:** {badge(row['Criterion_2'])}", unsafe_allow_html=True)

        st.markdown("---")

        # ── Evidence rates ──
        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Abstracts analyzed", int(row["Analyzed"]))
        ec2.metric("Rp (positive rate)", f"{row['Rate_Positive']:.1%}",
                   delta=f"{int(row['Positive'])} positive")
        ec3.metric("Ru (neutral rate)", f"{row['Rate_Neutral']:.1%}")
        ec4.metric("Rn (negative rate)", f"{row['Rate_Negative']:.1%}",
                   delta=f"{int(row['Negative'])} negative", delta_color="inverse")

        # ── Abstract Evidence Drill-down ──
        drug_abstracts = ab_df[ab_df["drug_name"] == selected] if not ab_df.empty else pd.DataFrame()

        if not drug_abstracts.empty:
            st.markdown("---")

            # Positive abstracts
            pos_abs = drug_abstracts[drug_abstracts["result"] == "Positive"]
            neg_abs = drug_abstracts[drug_abstracts["result"] == "Negative"]
            neu_abs = drug_abstracts[drug_abstracts["result"] == "Neutral"]

            st.markdown(f"#### 📑 Evidence sources — "
                        f"🟢 {len(pos_abs)} Positive · "
                        f"🔴 {len(neg_abs)} Negative · "
                        f"⚪ {len(neu_abs)} Neutral")

            # Show selector
            show_which = st.radio(
                "Show abstracts",
                ["🟢 Positive", "🔴 Negative", "⚪ Neutral", "All"],
                horizontal=True, index=0,
                key=f"abs_radio_{selected}"
            )

            if show_which == "🟢 Positive":
                show_abs = pos_abs
            elif show_which == "🔴 Negative":
                show_abs = neg_abs
            elif show_which == "⚪ Neutral":
                show_abs = neu_abs
            else:
                show_abs = drug_abstracts

            if show_abs.empty:
                st.info(f"No {show_which.split(' ')[1].lower()} abstracts for this drug.")
            else:
                for _, a in show_abs.iterrows():
                    result = a["result"]
                    pmid = str(a["pubmed_id"])
                    title = str(a.get("title", ""))
                    explanation = str(a.get("explanation", ""))
                    model = str(a.get("model", ""))

                    if result == "Positive":
                        css_class = "abstract-positive"
                        icon = "🟢"
                    elif result == "Negative":
                        css_class = "abstract-negative"
                        icon = "🔴"
                    else:
                        css_class = "abstract-neutral"
                        icon = "⚪"

                    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                    st.markdown(f"""
                    <div class="abstract-card {css_class}">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem;">
                            <span>{icon} <strong>{result}</strong>
                                &nbsp;·&nbsp; Model: {model}</span>
                            <a href="{pubmed_url}" target="_blank" class="pmid-link">PMID: {pmid} ↗</a>
                        </div>
                        <div style="font-weight:500; margin-bottom:0.3rem;">{title}</div>
                        <div style="color:#555; font-size:0.85rem;">{explanation if explanation != 'nan' else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Abstract-level data not loaded. Place `Copy_of_realtime_abstract_analysis__1_.csv` "
                    "in the app directory to enable drill-down.")

        # ── Clinical trials ──
        st.markdown("---")
        has_t = row["has_clinical_trial"] == "Yes"
        trial_count = int(row["clinical_trial_count"])
        st.markdown(f"#### Clinical trials ({trial_count} found)")
        if has_t:
            trial_ids = str(row.get("clinical_trial_id", ""))
            trial_title = str(row.get("trial_title", ""))
            trial_phase = str(row.get("study_phase", ""))
            trial_status = str(row.get("trial_status", ""))

            ids = [x.strip() for x in trial_ids.split("||")] if trial_ids and trial_ids != "nan" else []
            titles = [x.strip() for x in trial_title.split("||")] if trial_title and trial_title != "nan" else []
            phases = [x.strip() for x in trial_phase.split("||")] if trial_phase and trial_phase != "nan" else []
            statuses = [x.strip() for x in trial_status.split("||")] if trial_status and trial_status != "nan" else []

            trial_rows = []
            for i in range(len(ids)):
                trial_rows.append({
                    "NCT ID": ids[i] if i < len(ids) else "",
                    "Phase": phases[i] if i < len(phases) else "",
                    "Status": statuses[i] if i < len(statuses) else "",
                    "Title": titles[i] if i < len(titles) else "",
                })
            if trial_rows:
                st.dataframe(pd.DataFrame(trial_rows), use_container_width=True, hide_index=True)
        else:
            st.success("🌟 **Novel candidate** — no existing AIC-related clinical trials found.")

        # Cross-model comparison
        if len(rows) > 1:
            st.markdown("#### Cross-model comparison")
            compare_cols = ["Model", "rank", "Analyzed", "Positive", "Neutral", "Negative",
                           "Rate_Positive", "Criterion_1", "Criterion_2"]
            st.dataframe(rows[compare_cols], use_container_width=True, hide_index=True)


# ─── TAB 4: NOVEL CANDIDATES ────────────────────────────────────
with tab_novel:
    st.markdown("### 🌟 Novel candidates — Therapeutic (C2) with no clinical trials")
    st.markdown(
        "Drugs with positive literature signals but **no AIC-related clinical trials**. "
        "This filter identified **Cysteine, Dasatinib, Tranilast, and Tretinoin** — "
        "all subsequently validated in vivo."
    )

    novel = filt[
        (filt["Criterion_2"] == "Potentially Therapeutic") & (~filt["has_trial"])
    ].sort_values("Rate_Positive", ascending=False)

    if novel.empty:
        st.info("No novel candidates match current filters.")
    else:
        for _, r in novel.iterrows():
            is_val = r["drug_name"] in VALIDATED_DRUGS
            border_color = "#FFD600" if is_val else "#4CAF50"
            v_badge = (' <span class="validated-badge">✓ VALIDATED IN VIVO</span>'
                       if is_val else "")
            bg = ('linear-gradient(135deg, #FFFDE7 0%, #FFF9C4 100%)' if is_val
                  else 'linear-gradient(135deg, #E8F5E9 0%, #F1F8E9 100%)')

            # Get positive PMIDs for this drug
            pmid_links = ""
            if not ab_df.empty:
                pos_for_drug = ab_df[(ab_df["drug_name"] == r["drug_name"]) & (ab_df["result"] == "Positive")]
                if not pos_for_drug.empty:
                    links = []
                    for _, pa in pos_for_drug.iterrows():
                        pid = str(pa["pubmed_id"])
                        links.append(f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pid}/" '
                                     f'target="_blank" class="pmid-link">{pid}</a>')
                    pmid_links = f'<div style="margin-top:0.4rem; font-size:0.82rem; color:#666;">Positive evidence PMIDs: {" · ".join(links)}</div>'

            st.markdown(f"""
            <div style="border: 2px solid {border_color}; border-radius: 10px;
                        padding: 1rem; margin: 0.5rem 0; background: {bg};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="font-size: 1.1rem;">{r['drug_name']}</strong>{v_badge}
                        &nbsp;&nbsp;
                        <span style="color: #666;">({r['Model']}, rank #{int(r['rank']) if r['rank']>0 else 'N/A'})</span>
                    </div>
                    <div>{badge(r['Criterion_1'])} &nbsp; {badge(r['Criterion_2'])}</div>
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #555;">
                    Analyzed: <strong>{int(r['Analyzed'])}</strong> abstracts &nbsp;|&nbsp;
                    Positive: <strong>{int(r['Positive'])}</strong> ({r['Rate_Positive']:.1%}) &nbsp;|&nbsp;
                    Neutral: <strong>{int(r['Neutral'])}</strong> &nbsp;|&nbsp;
                    Negative: <strong>{int(r['Negative'])}</strong>
                </div>
                {pmid_links}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        n_validated = len(novel[novel["validated_in_paper"]])
        st.markdown(
            f"**{len(novel)} novel candidates** identified. "
            f"**{n_validated}** were validated in vivo "
            f"(Dasatinib via eAIC zebrafish; Cysteine, Tranilast, Tretinoin via aAIC zebrafish)."
        )


# ── Footer ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #999; font-size: 0.8rem;'>"
    "PriorRx: AI-Prioritized Drug Repurposing Candidates Platform &nbsp;·&nbsp; "
    "Union-then-Filter Framework &nbsp;·&nbsp; "
    "TxGNN + CompGCN + RLR on PrimeKG &nbsp;·&nbsp; "
    "LLM: GPT-4.1 &nbsp;·&nbsp; "
    "© 2025 TaosLab"
    "</div>",
    unsafe_allow_html=True
)
