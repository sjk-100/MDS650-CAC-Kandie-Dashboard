"""
MDS650 — Bayesian CAC Optimiser Dashboard
Run:  streamlit run app.py
Requires: streamlit, joblib, numpy, pandas, scipy, scikit-learn, plotly
Place this file in the same folder as your /outputs/ directory.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import differential_evolution
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CAC Optimiser · MDS650",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants matching the notebook ──────────────────────────────────────────
CHANNELS = ["Meta", "Google", "TikTok"]
CPM_AUD  = {"Meta": 27.59, "Google": 30.55, "TikTok": 8.29}
AOV_AUD  = {"Meta": 94.78, "Google": 206.01, "TikTok": 51.38}
LPV_RATE = {"Meta": 0.90,  "Google": 0.92,   "TikTok": 0.85}
ATC_RATE = {"Meta": 0.12,  "Google": 0.15,   "TikTok": 0.10}
CHANNEL_COLORS = {"Meta": "#1B3A6B", "Google": "#2E6E62", "TikTok": "#E8A838"}

# ── Load artefacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.path.join("best_cac_model.joblib")
    return joblib.load(path)

@st.cache_data
def load_scenarios():
    return pd.read_csv(os.path.join("monte_carlo_scenarios.csv"))

@st.cache_data
def load_sensitivity():
    return pd.read_csv(os.path.join("sensitivity_analysis.csv"))

@st.cache_data
def load_model_eval():
    return pd.read_csv(os.path.join("model_evaluation.csv"))

@st.cache_data
def load_allocation():
    return pd.read_csv(os.path.join("optimal_channel_allocation.csv"))

try:
    model       = load_model()
    scenarios   = load_scenarios()
    sensitivity = load_sensitivity()
    model_eval  = load_model_eval()
    allocation  = load_allocation()
    data_ok = True
except FileNotFoundError as e:
    data_ok = False
    missing = str(e)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Budget Settings")
    total_budget = st.number_input(
        "Total weekly budget (AUD $)",
        min_value=1_000, max_value=100_000,
        value=10_000, step=500,
        help="Total advertising spend to allocate across Meta, Google Ads, and TikTok"
    )

    st.markdown("---")
    st.markdown("## 🎛️ Manual Allocation")
    st.caption("Adjust sliders to explore a custom split")

    meta_pct   = st.slider("Meta (%)",   5, 90, 33)
    google_pct = st.slider("Google (%)", 5, 90, 34)
    tiktok_pct = 100 - meta_pct - google_pct
    tiktok_pct = max(5, min(90, tiktok_pct))

    if meta_pct + google_pct + tiktok_pct != 100:
        remaining = 100 - meta_pct - google_pct
        st.warning(f"TikTok auto-set to {remaining}% so total = 100%")

    st.metric("TikTok (%)", tiktok_pct)

    st.markdown("---")
    st.caption("MDS650 Data Science Capstone · Sydney Polytechnic Institute")

# ── Error gate ────────────────────────────────────────────────────────────────
if not data_ok:
    st.error(f"Could not find outputs folder. Run the notebook first.\n\n`{missing}`")
    st.info("Expected files: outputs/best_cac_model.joblib, "
            "monte_carlo_scenarios.csv, model_evaluation.csv, "
            "optimal_channel_allocation.csv, sensitivity_analysis.csv")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-family:sans-serif;color:#1B3A6B;margin-bottom:4px'>
    CAC Prediction & Channel Optimiser
</h1>
<p style='color:#555;font-size:1rem;margin-top:0'>
    Bayesian Multi-Channel Attribution Framework · Australian DTC Health Supplement Market
</p>
""", unsafe_allow_html=True)
st.divider()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Optimiser",
    "🔍 Manual Prediction",
    "📊 Simulation Explorer",
    "🌡️ Sensitivity Analysis"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OPTIMISER
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Optimal Channel Allocation")
    st.caption(
        "Differential Evolution finds the budget split that minimises "
        "predicted blended CAC at your chosen budget."
    )

    run_opt = st.button("▶  Run Optimiser", type="primary", use_container_width=True)

    if run_opt:
        with st.spinner("Running Differential Evolution optimiser…"):
            def objective_de(x):
                shares = np.clip(x, 0.05, 0.80)
                shares = shares / shares.sum()
                feat   = np.array([[
                    shares[0] * total_budget,
                    shares[1] * total_budget,
                    shares[2] * total_budget,
                    1.0, 1.0, 1.0
                ]])
                return model.predict(feat)[0]

            de_result = differential_evolution(
                objective_de,
                bounds=[(0.05, 0.80)] * 3,
                seed=42, maxiter=500, tol=1e-10,
                popsize=20, mutation=(0.5, 1), recombination=0.7, workers=1
            )
            opt = np.clip(de_result.x, 0.05, 0.80)
            opt = opt / opt.sum()

        pred_cac = de_result.fun

        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Meta",       f"{opt[0]*100:.1f}%", f"AUD ${opt[0]*total_budget:,.0f}")
        c2.metric("Google Ads", f"{opt[1]*100:.1f}%", f"AUD ${opt[1]*total_budget:,.0f}")
        c3.metric("TikTok",     f"{opt[2]*100:.1f}%", f"AUD ${opt[2]*total_budget:,.0f}")
        c4.metric("Predicted blended CAC", f"AUD ${pred_cac:.2f}")

        st.divider()

        # Donut chart
        fig_donut = go.Figure(go.Pie(
            labels=CHANNELS,
            values=[opt[0]*total_budget, opt[1]*total_budget, opt[2]*total_budget],
            hole=0.55,
            marker_colors=[CHANNEL_COLORS[c] for c in CHANNELS],
            textinfo="label+percent",
            textfont_size=14,
        ))
        fig_donut.update_layout(
            title=f"Optimal allocation of AUD ${total_budget:,}",
            showlegend=False, height=380,
            margin=dict(t=50, b=20, l=0, r=0)
        )

        # CAC surface across Google share
        g_shares = np.linspace(0.05, 0.80, 60)
        cacs     = []
        for gs in g_shares:
            rem  = (1 - gs) / 2
            feat = np.array([[rem*total_budget, gs*total_budget,
                              rem*total_budget, 1., 1., 1.]])
            cacs.append(model.predict(feat)[0])

        fig_surface = go.Figure()
        fig_surface.add_trace(go.Scatter(
            x=g_shares * 100, y=cacs,
            mode="lines", line=dict(color="#2E6E62", width=3),
            name="Predicted CAC"
        ))
        fig_surface.add_vline(x=opt[1]*100, line_dash="dash",
            line_color="#E8A838", line_width=2,
            annotation_text=f"Optimal {opt[1]*100:.1f}%",
            annotation_position="top right")
        fig_surface.add_vline(x=33.3, line_dash="dot",
            line_color="#1B3A6B", line_width=1.5,
            annotation_text="Equal split",
            annotation_position="top left")
        fig_surface.update_layout(
            title="CAC Surface Across Google Ads Share<br>"
                  "<sup>Meta and TikTok split equally on remaining budget</sup>",
            xaxis_title="Google Ads Share (%)",
            yaxis_title="Predicted Blended CAC (AUD)",
            height=380,
            margin=dict(t=70, b=50, l=60, r=20)
        )

        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(fig_donut, use_container_width=True)
        with col_r:
            st.plotly_chart(fig_surface, use_container_width=True)

    else:
        # Show previously saved optimal allocation
        st.info("Press **Run Optimiser** to compute the optimal allocation at "
                f"AUD ${total_budget:,}. Or view the saved result below from the notebook run.")
        opt_pct = allocation["allocation_percent"].values
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Meta",       f"{opt_pct[0]:.1f}%")
        c2.metric("Google Ads", f"{opt_pct[1]:.1f}%")
        c3.metric("TikTok",     f"{opt_pct[2]:.1f}%")
        pred_saved = model.predict([[
            allocation["recommended_budget_aud"].values[0],
            allocation["recommended_budget_aud"].values[1],
            allocation["recommended_budget_aud"].values[2],
            1., 1., 1.
        ]])[0]
        c4.metric("Predicted CAC (saved run)", f"AUD ${pred_saved:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MANUAL PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Manual Allocation Predictor")
    st.caption("Uses the sidebar sliders. Adjust the split to explore different scenarios.")

    shares  = np.array([meta_pct, google_pct, tiktok_pct]) / 100
    spends  = shares * total_budget
    feat    = np.array([[spends[0], spends[1], spends[2], 1., 1., 1.]])
    pred    = model.predict(feat)[0]

    c1, c2, c3 = st.columns(3)
    for col, ch, sp in zip([c1, c2, c3], CHANNELS, spends):
        col.metric(f"{ch} spend", f"AUD ${sp:,.0f}", f"{sp/total_budget*100:.1f}%")

    st.divider()

    c4, c5 = st.columns([1, 2])
    with c4:
        st.metric("Predicted blended CAC", f"AUD ${pred:.2f}",
                  help="Predicted by the Random Forest model at neutral seasonality/adstock/saturation")
        sustainability = total_budget / pred
        st.metric("Estimated weekly customers", f"{sustainability:.1f}")
        ltv_threshold = 400
        ratio = ltv_threshold / pred
        color = "green" if ratio >= 3 else "orange" if ratio >= 2 else "red"
        st.markdown(f"**LTV:CAC ratio** (at AUD ${ltv_threshold} LTV):  "
                    f"<span style='color:{color};font-weight:bold'>{ratio:.1f}x</span>",
                    unsafe_allow_html=True)
        if ratio >= 3:
            st.success("Above 3:1 sustainability threshold ✓")
        elif ratio >= 2:
            st.warning("Below ideal 3:1 — monitor closely")
        else:
            st.error("Below 2:1 — allocation not commercially viable")

    with c5:
        # Compare to optimised
        opt_cac = model.predict([[
            allocation["recommended_budget_aud"].values[0],
            allocation["recommended_budget_aud"].values[1],
            allocation["recommended_budget_aud"].values[2],
            1., 1., 1.
        ]])[0]

        fig_compare = go.Figure()
        scenarios_compare = {
            "Your allocation": pred,
            "DE-Optimal":      opt_cac,
            "Equal split":     model.predict([[total_budget/3, total_budget/3,
                                               total_budget/3, 1., 1., 1.]])[0],
        }
        colors = ["#1B3A6B", "#2E6E62", "#999999"]
        fig_compare.add_trace(go.Bar(
            x=list(scenarios_compare.keys()),
            y=list(scenarios_compare.values()),
            marker_color=colors,
            text=[f"AUD ${v:.2f}" for v in scenarios_compare.values()],
            textposition="outside"
        ))
        fig_compare.update_layout(
            title="Your Allocation vs Benchmarks",
            yaxis_title="Predicted Blended CAC (AUD)",
            yaxis_range=[0, max(scenarios_compare.values()) * 1.3],
            height=360,
            margin=dict(t=50, b=40, l=60, r=20)
        )
        st.plotly_chart(fig_compare, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SIMULATION EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Monte Carlo Simulation Explorer")
    st.caption(f"{len(scenarios):,} scenarios — each row is one simulated weekly campaign")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median CAC",  f"AUD ${scenarios['blended_cac_aud'].median():.2f}")
    c2.metric("Mean CAC",    f"AUD ${scenarios['blended_cac_aud'].mean():.2f}")
    c3.metric("5th pct CAC", f"AUD ${scenarios['blended_cac_aud'].quantile(0.05):.2f}")
    c4.metric("95th pct CAC",f"AUD ${scenarios['blended_cac_aud'].quantile(0.95):.2f}")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        # CAC distribution histogram
        fig_hist = px.histogram(
            scenarios, x="blended_cac_aud",
            nbins=60, title="Blended CAC Distribution (all 9,999 scenarios)",
            labels={"blended_cac_aud": "Blended CAC (AUD)"},
            color_discrete_sequence=["#2E6E62"]
        )
        fig_hist.add_vline(x=scenarios["blended_cac_aud"].median(),
            line_dash="dash", line_color="#E8A838", line_width=2,
            annotation_text="Median", annotation_position="top right")
        fig_hist.update_layout(height=380, margin=dict(t=50,b=50,l=60,r=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_r:
        # Scatter: Google spend vs CAC
        sample = scenarios.sample(min(2000, len(scenarios)), random_state=42)
        fig_scatter = px.scatter(
            sample, x="google_spend", y="blended_cac_aud",
            color="meta_spend",
            color_continuous_scale="Blues",
            title="Google Spend vs Blended CAC<br><sup>Colour = Meta spend</sup>",
            labels={"google_spend": "Google Spend (AUD)",
                    "blended_cac_aud": "Blended CAC (AUD)",
                    "meta_spend": "Meta Spend"},
            opacity=0.4, height=380
        )
        fig_scatter.update_layout(margin=dict(t=70,b=50,l=60,r=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Model Performance")
    st.dataframe(
        model_eval.style.format({"RMSE": "${:.2f}", "MAE": "${:.2f}", "R2": "{:.3f}"}),
        use_container_width=True
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Sensitivity Analysis")
    st.caption("How median CAC changes when each model assumption is varied ±20%")

    dimensions = sensitivity["Dimension"].unique().tolist()
    selected   = st.selectbox("Select dimension to plot", dimensions)

    subset = sensitivity[sensitivity["Dimension"] == selected].copy()

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=subset["Factor"], y=subset["Median CAC"],
        mode="lines+markers",
        line=dict(color="#1B3A6B", width=3),
        marker=dict(size=8, color="#2E6E62"),
        name="Median CAC"
    ))
    fig_sens.add_trace(go.Scatter(
        x=subset["Factor"], y=subset["p95 CAC"],
        mode="lines", line=dict(color="#E8A838", dash="dash"),
        name="95th pct CAC"
    ))
    fig_sens.add_trace(go.Scatter(
        x=subset["Factor"], y=subset["p5 CAC"],
        mode="lines", line=dict(color="#E8A838", dash="dot"),
        name="5th pct CAC",
        fill="tonexty", fillcolor="rgba(232,168,56,0.08)"
    ))
    fig_sens.update_layout(
        title=f"Sensitivity: {selected}",
        xaxis_title="Factor",
        yaxis_title="CAC (AUD)",
        height=420,
        margin=dict(t=60,b=60,l=60,r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_sens, use_container_width=True)

    st.subheader("Full sensitivity table")
    st.dataframe(
        sensitivity.style.format({
            "Median CAC": "${:,.2f}", "Mean CAC": "${:,.2f}",
            "p5 CAC": "${:,.2f}",   "p95 CAC": "${:,.2f}"
        }),
        use_container_width=True, height=420
    )
