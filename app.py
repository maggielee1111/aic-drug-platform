"""
AIC Drug Repurposing Platform — Interactive Dashboard
=====================================================
Reads directly from clinicaltrials_cardiomyopathy_drugs_results_aggregated.csv
(production output from your LLM pipeline).

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="AIC Drug Repurposing Platform",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1B3A5C 0%, #2A7F8E 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p { color: #B0D4E8; margin: 0.3rem 0 0 0; font-size: 0.95rem; }
    .badge-positive { background: #E8F5E9; color: #2E7D32; padding: 3px 10px; border-radius: 4px; font-weight: 600; }
    .badge-adverse  { background: #FFEBEE; color: #C62828; padding: 3px 10px; border-radius: 4px; font-weight: 600; }
    .badge-neutral  { background: #FFF3E0; color: #E65100; padding: 3px 10px; border-radius: 4px; font-weight: 600; }
    .badge-nosign   { background: #ECEFF1; color: #546E7A; padding: 3px 10px; border-radius: 4px; font-weight: 600; }
    .novel-card {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #E8F5E9 0%, #F1F8E9 100%);
    }
    .validated-badge {
        background: #FFD600; color: #333; padding: 2px 8px; border-radius: 4px;
        font-weight: 700; font-size: 0.75rem; vertical-align: middle;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Load data ────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__),
                        "clinicaltrials_cardiomyopathy_drugs_results_aggregated.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)

    # De-duplicate: keep first occurrence per (drug_name, Model)
    df = df.drop_duplicates(subset=["drug_name", "Model"], keep="first")

    # Fill NaN study-type columns with 0
    study_cols = ["Human_Studies", "Animal_in_vivo_Studies", "In_vitro_Studies",
                  "Review_Studies", "Computational_Studies", "Not_specified_Studies"]
    for c in study_cols:
        df[c] = df[c].fillna(0).astype(int)

    # Clean rank
    df["rank"] = df["rank"].fillna(0).astype(int)
    df["rank_learnprime"] = df["rank_learnprime"].fillna(0).astype(int)

    # Boolean flag
    df["has_trial"] = df["has_clinical_trial"] == "Yes"

    # Count number of models each drug appears in
    model_counts = df.groupby("drug_name")["Model"].nunique().rename("n_models")
    df = df.merge(model_counts, on="drug_name", how="left")

    # Validated in vivo drugs from AIC paper
    df["validated_in_paper"] = df["drug_name"].isin(
        ["Cysteine", "Dasatinib", "Tranilast", "Tretinoin"]
    )

    return df

df = load_data()


# ── Helper ───────────────────────────────────────────────────────
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

VALIDATED_DRUGS = {"Cysteine", "Dasatinib", "Tranilast", "Tretinoin"}


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 Filters")

    models = st.multiselect(
        "Graph model source",
        sorted(df["Model"].unique()),
        default=sorted(df["Model"].unique()),
    )

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
    trial_opt = st.radio("Show drugs with trials", ["All", "Has trials", "No trials (novel)"],
                         index=0)

    st.markdown("---")
    search = st.text_input("🔎 Search drug name", "")

    show_validated = st.checkbox("Highlight validated drugs", value=True)


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
    <h1>💊 AIC Drug Repurposing Platform</h1>
    <p>LLM-guided prioritization of therapeutics for Anthracycline-Induced Cardiotoxicity
    &nbsp;·&nbsp; Union-then-Filter framework &nbsp;·&nbsp; TxGNN + CompGCN + RLR on PrimeKG</p>
</div>
""", unsafe_allow_html=True)

# ── Metrics ──────────────────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Total entries", len(df))
m2.metric("Showing", len(filt))
m3.metric("Unique drugs", df["drug_name"].nunique())
m4.metric("Therapeutic (C1)", len(filt[filt["Criterion_1"] == "Potentially Therapeutic"]))
m5.metric("Therapeutic (C2)", len(filt[filt["Criterion_2"] == "Potentially Therapeutic"]))
novel_count = len(filt[(filt["Criterion_2"] == "Potentially Therapeutic") & (~filt["has_trial"])])
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

    display = filt[[
        "drug_name", "Model", "rank", "Criterion_1", "Criterion_2",
        "Analyzed", "Positive", "Neutral", "Negative",
        "Rate_Positive", "Rate_Negative",
        "Human_Studies", "Animal_in_vivo_Studies", "In_vitro_Studies",
        "Review_Studies", "Computational_Studies",
        "has_clinical_trial", "clinical_trial_count",
    ]].copy()

    display.columns = [
        "Drug", "Model", "Rank", "Criterion 1", "Criterion 2",
        "Analyzed", "Pos", "Neu", "Neg",
        "Rp", "Rn",
        "Human", "Animal", "In vitro",
        "Review", "Comput.",
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
        .applymap(color_c1, subset=["Criterion 1"])\
        .applymap(color_c2, subset=["Criterion 2"])\
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
                           legend=dict(orientation="h", y=-0.2))
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
                           legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig2, use_container_width=True)

    # Evidence type pie (only rows with study type data)
    st.markdown("#### Study type distribution across all abstracts")
    study_sums = filt[["Human_Studies", "Animal_in_vivo_Studies", "In_vitro_Studies",
                       "Review_Studies", "Computational_Studies", "Not_specified_Studies"]].sum()
    study_labels = ["Human", "Animal in vivo", "In vitro", "Review", "Computational", "Not specified"]
    if study_sums.sum() > 0:
        fig3 = px.pie(values=study_sums.values, names=study_labels,
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig3.update_layout(height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Study type data not yet available for the selected drugs. "
                "Run `2.1llm_abstract_analysis_with_study_type.py` to populate.")

    # Scatter: score vs. positive rate
    st.markdown("#### Positive rate (Rp) by model rank")
    scatter = filt[filt["rank"] > 0].copy()
    if not scatter.empty:
        scatter["label"] = scatter["drug_name"]
        scatter.loc[~scatter["validated_in_paper"], "label"] = ""
        fig4 = px.scatter(scatter, x="rank", y="Rate_Positive",
                          color="Model", size="Analyzed",
                          hover_name="drug_name",
                          text="label",
                          labels={"rank": "Model rank (lower = higher priority)",
                                  "Rate_Positive": "Positive rate (Rp)",
                                  "Analyzed": "Abstracts analyzed"},
                          color_discrete_sequence=["#1B3A5C", "#2A7F8E", "#E8734A"])
        fig4.update_traces(textposition="top center", textfont_size=11)
        fig4.update_layout(height=420, margin=dict(t=10, b=30))
        st.plotly_chart(fig4, use_container_width=True)


# ─── TAB 3: DRUG DETAIL ─────────────────────────────────────────
with tab_detail:
    st.markdown("### 🔬 Drug detail view")

    drug_names = sorted(filt["drug_name"].unique())
    if not drug_names:
        st.warning("No drugs match current filters.")
    else:
        # Pre-select a validated drug if available
        default_idx = 0
        for i, d in enumerate(drug_names):
            if d in VALIDATED_DRUGS:
                default_idx = i
                break
        selected = st.selectbox("Select a drug", drug_names, index=default_idx)

        rows = filt[filt["drug_name"] == selected]
        row = rows.iloc[0]

        # Header
        hc1, hc2, hc3 = st.columns([2, 1, 1])
        with hc1:
            title_extra = ""
            if row["validated_in_paper"]:
                title_extra = ' <span class="validated-badge">✓ VALIDATED IN VIVO</span>'
            st.markdown(f"## {selected}{title_extra}", unsafe_allow_html=True)
            if len(rows) > 1:
                models_str = ", ".join(rows["Model"].tolist())
                st.markdown(f"Appears in **{len(rows)} models**: {models_str}")
        with hc2:
            st.markdown(f"**Model:** {row['Model']}")
            st.markdown(f"**Rank:** #{row['rank']}" if row['rank'] > 0 else "**Rank:** N/A")
        with hc3:
            st.markdown(f"**Criterion 1:** {badge(row['Criterion_1'])}", unsafe_allow_html=True)
            st.markdown(f"**Criterion 2:** {badge(row['Criterion_2'])}", unsafe_allow_html=True)

        st.markdown("---")

        # Evidence rates
        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Abstracts analyzed", int(row["Analyzed"]))
        ec2.metric("Rp (positive rate)", f"{row['Rate_Positive']:.1%}",
                   delta=f"{int(row['Positive'])} positive")
        ec3.metric("Ru (neutral rate)", f"{row['Rate_Neutral']:.1%}")
        ec4.metric("Rn (negative rate)", f"{row['Rate_Negative']:.1%}",
                   delta=f"{int(row['Negative'])} negative", delta_color="inverse")

        # Study type breakdown
        st.markdown("#### Study type breakdown")
        study_data = {
            "Human": int(row["Human_Studies"]),
            "Animal in vivo": int(row["Animal_in_vivo_Studies"]),
            "In vitro": int(row["In_vitro_Studies"]),
            "Review": int(row["Review_Studies"]),
            "Computational": int(row["Computational_Studies"]),
            "Not specified": int(row["Not_specified_Studies"]),
        }
        if sum(study_data.values()) > 0:
            fig_bar = go.Figure(go.Bar(
                x=list(study_data.keys()),
                y=list(study_data.values()),
                marker_color=["#2E7D32", "#66BB6A", "#81C784", "#A5D6A7", "#C8E6C9", "#E0E0E0"],
                text=list(study_data.values()),
                textposition="outside",
            ))
            fig_bar.update_layout(height=260, margin=dict(t=10, b=30, l=40, r=10),
                                  yaxis_title="Abstract count",
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Study type data not available — run script 2.1 for this drug.")

        # Clinical trials
        has_t = row["has_clinical_trial"] == "Yes"
        trial_count = int(row["clinical_trial_count"])
        st.markdown(f"#### Clinical trials ({trial_count} found)")
        if has_t:
            trial_ids = str(row.get("clinical_trial_id", ""))
            trial_title = str(row.get("trial_title", ""))
            trial_phase = str(row.get("study_phase", ""))
            trial_status = str(row.get("trial_status", ""))
            trial_conditions = str(row.get("trial_conditions", ""))

            # Parse multiple trials (separated by ||)
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
                trial_df = pd.DataFrame(trial_rows)
                st.dataframe(trial_df, use_container_width=True, hide_index=True)

            if row.get("additional_trials") and str(row["additional_trials"]) != "nan":
                st.caption(f"Additional trials: {row['additional_trials']}")
        else:
            st.success(
                "🌟 **Novel candidate** — no existing AIC-related clinical trials found. "
                "This drug may represent a new repurposing opportunity."
            )

        # If drug appears in multiple models, show comparison
        if len(rows) > 1:
            st.markdown("#### Cross-model comparison")
            compare_cols = ["Model", "rank", "Analyzed", "Positive", "Neutral", "Negative",
                           "Rate_Positive", "Criterion_1", "Criterion_2"]
            st.dataframe(rows[compare_cols], use_container_width=True, hide_index=True)


# ─── TAB 4: NOVEL CANDIDATES ────────────────────────────────────
with tab_novel:
    st.markdown("### 🌟 Novel candidates — Potentially Therapeutic (C2) with no clinical trials")
    st.markdown(
        "These drugs have positive evidence signals in the literature but have **not been tested** "
        "in AIC-related clinical trials. This is the exact filter that identified **Cysteine, "
        "Dasatinib, Tranilast, and Tretinoin** in the original study."
    )

    novel = filt[
        (filt["Criterion_2"] == "Potentially Therapeutic") & (~filt["has_trial"])
    ].sort_values("Rate_Positive", ascending=False)

    if novel.empty:
        st.info("No novel candidates match current filters.")
    else:
        for _, r in novel.iterrows():
            is_validated = r["drug_name"] in VALIDATED_DRUGS
            border_color = "#FFD600" if is_validated else "#4CAF50"
            v_badge = (' <span class="validated-badge">✓ VALIDATED IN VIVO</span>'
                       if is_validated else "")

            st.markdown(f"""
            <div style="border: 2px solid {border_color}; border-radius: 10px;
                        padding: 1rem; margin: 0.5rem 0;
                        background: {'linear-gradient(135deg, #FFFDE7 0%, #FFF9C4 100%)' if is_validated
                                      else 'linear-gradient(135deg, #E8F5E9 0%, #F1F8E9 100%)'} ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="font-size: 1.1rem;">{r['drug_name']}</strong>{v_badge}
                        &nbsp;&nbsp;
                        <span style="color: #666;">({r['Model']}, rank #{int(r['rank']) if r['rank']>0 else 'N/A'})</span>
                    </div>
                    <div>
                        {badge(r['Criterion_1'])} &nbsp; {badge(r['Criterion_2'])}
                    </div>
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #555;">
                    Analyzed: <strong>{int(r['Analyzed'])}</strong> abstracts &nbsp;|&nbsp;
                    Positive: <strong>{int(r['Positive'])}</strong> ({r['Rate_Positive']:.1%}) &nbsp;|&nbsp;
                    Neutral: <strong>{int(r['Neutral'])}</strong> &nbsp;|&nbsp;
                    Negative: <strong>{int(r['Negative'])}</strong>
                    {"&nbsp;|&nbsp; Human: <strong>" + str(int(r['Human_Studies'])) + "</strong> &nbsp;|&nbsp; Animal: <strong>" + str(int(r['Animal_in_vivo_Studies'])) + "</strong>" if r['Human_Studies'] > 0 or r['Animal_in_vivo_Studies'] > 0 else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            f"**{len(novel)} novel candidates** identified. "
            f"Of these, **{len(novel[novel['validated_in_paper']])}** were subsequently validated "
            f"in vivo (Dasatinib via eAIC zebrafish model; Cysteine, Tranilast, Tretinoin via aAIC zebrafish model)."
        )


# ── Footer ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #999; font-size: 0.8rem;'>"
    "AIC Drug Repurposing Platform &nbsp;·&nbsp; Union-then-Filter Framework &nbsp;·&nbsp; "
    "Data: 89 drug entries from TxGNN + CompGCN + RLR on PrimeKG &nbsp;·&nbsp; "
    "LLM: GPT-4.1 (context-only prompting)"
    "</div>",
    unsafe_allow_html=True
)
