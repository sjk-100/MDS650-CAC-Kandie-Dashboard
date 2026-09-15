import streamlit as st
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import differential_evolution
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="CAC Optimiser · MDS650",
                   page_icon="📊", layout="wide")

CHANNELS     = ["Meta", "Google", "TikTok"]
FEATURE_COLS = ["meta_spend","google_spend","tiktok_spend",
                "seasonality_index","adstock","saturation"]

def feat(m,g,t): 
    return pd.DataFrame([[m,g,t,1.,1.,1.]], columns=FEATURE_COLS)

@st.cache_resource
def load_model():
    return joblib.load("best_cac_model.joblib")

@st.cache_data
def load_csv(path):
    try: return pd.read_csv(path)
    except: return None

model     = load_model()
scenarios = load_csv("monte_carlo_scenarios.csv")
model_eval= load_csv("model_evaluation.csv")
allocation= load_csv("optimal_channel_allocation.csv")
sensitivity=load_csv("sensitivity_analysis.csv")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 CAC Prediction & Channel Optimiser")
st.caption("Bayesian Multi-Channel Attribution · Australian DTC Health Supplement Market")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    budget = st.number_input("Total weekly budget (AUD $)",
                             1000, 100000, 10000, 500)
    st.markdown("---")
    st.subheader("🎛️ Manual split")
    meta_p   = st.slider("Meta (%)",   5, 90, 6)
    google_p = st.slider("Google (%)", 5, 90, 89)
    tiktok_p = max(5, 100 - meta_p - google_p)
    st.metric("TikTok (%)", tiktok_p)

tab1,tab2,tab3,tab4 = st.tabs([
    "📈 Optimiser","🔍 Manual","📊 Simulation","🌡️ Sensitivity"])

# ── TAB 1 — Optimiser ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("Optimal Channel Allocation")
    if st.button("▶ Run Optimiser", type="primary", use_container_width=True):
        with st.spinner("Running Differential Evolution…"):
            def obj(x):
                s = np.clip(x,0.05,0.80); s=s/s.sum()
                return model.predict(feat(s[0]*budget,s[1]*budget,s[2]*budget))[0]
            res = differential_evolution(obj,[(0.05,0.80)]*3,
                  seed=42,maxiter=500,tol=1e-10,popsize=20,workers=1)
            opt = np.clip(res.x,0.05,0.80); opt=opt/opt.sum()

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Meta",       f"{opt[0]*100:.1f}%",f"AUD ${opt[0]*budget:,.0f}")
        c2.metric("Google Ads", f"{opt[1]*100:.1f}%",f"AUD ${opt[1]*budget:,.0f}")
        c3.metric("TikTok",     f"{opt[2]*100:.1f}%",f"AUD ${opt[2]*budget:,.0f}")
        c4.metric("Predicted CAC", f"AUD ${res.fun:.2f}")
        st.divider()

        col_l,col_r = st.columns(2)
        with col_l:
            fig = go.Figure(go.Pie(
                labels=CHANNELS,
                values=[opt[0]*budget,opt[1]*budget,opt[2]*budget],
                hole=0.55,
                marker_colors=["#1B3A6B","#2E6E62","#E8A838"],
                textinfo="label+percent"))
            fig.update_layout(title=f"Optimal split — AUD ${budget:,}",
                              showlegend=False,height=360)
            st.plotly_chart(fig,use_container_width=True)

        with col_r:
            gs   = np.linspace(0.05,0.80,60)
            cacs = [model.predict(feat((1-g)/2*budget,g*budget,(1-g)/2*budget))[0]
                    for g in gs]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=gs*100,y=cacs,mode="lines",
                           line=dict(color="#2E6E62",width=3)))
            fig2.add_vline(x=opt[1]*100,line_dash="dash",line_color="#E8A838",
                           annotation_text=f"Optimal {opt[1]*100:.0f}%")
            fig2.add_vline(x=33.3,line_dash="dot",line_color="#1B3A6B",
                           annotation_text="Equal split")
            fig2.update_layout(title="CAC Surface — Google Ads Share",
                               xaxis_title="Google Ads Share (%)",
                               yaxis_title="Predicted CAC (AUD)",height=360)
            st.plotly_chart(fig2,use_container_width=True)
    else:
        st.info("Press **Run Optimiser** to find the optimal allocation.")

# ── TAB 2 — Manual ───────────────────────────────────────────────────────────
with tab2:
    st.subheader("Manual Allocation Predictor")
    shares = np.array([meta_p,google_p,tiktok_p])/100
    spends = shares*budget
    pred   = model.predict(feat(*spends))[0]

    c1,c2,c3 = st.columns(3)
    for col,ch,sp in zip([c1,c2,c3],CHANNELS,spends):
        col.metric(f"{ch}",f"AUD ${sp:,.0f}",f"{sp/budget*100:.1f}%")
    st.divider()

    c4,c5 = st.columns([1,2])
    with c4:
        st.metric("Predicted CAC",f"AUD ${pred:.2f}")
        ratio = 400/pred if pred>0 else 0
        st.metric("LTV:CAC (at AUD $400 LTV)",f"{ratio:.1f}×")
        if ratio>=3:   st.success("Above 3:1 threshold ✓")
        elif ratio>=2: st.warning("Below 3:1 — monitor")
        else:          st.error("Below 2:1 — not viable")
    with c5:
        opt_c = model.predict(feat(0.056*budget,0.888*budget,0.056*budget))[0]
        eq_c  = model.predict(feat(budget/3,budget/3,budget/3))[0]
        fig3  = go.Figure(go.Bar(
            x=["Your split","DE-Optimal","Equal split"],
            y=[pred,opt_c,eq_c],
            marker_color=["#1B3A6B","#2E6E62","#999"],
            text=[f"${v:.0f}" for v in [pred,opt_c,eq_c]],
            textposition="outside"))
        fig3.update_layout(yaxis_title="Predicted CAC (AUD)",
                           yaxis_range=[0,max(pred,opt_c,eq_c)*1.35],height=340)
        st.plotly_chart(fig3,use_container_width=True)

# ── TAB 3 — Simulation ────────────────────────────────────────────────────────
with tab3:
    st.subheader("Monte Carlo Simulation Explorer")
    if scenarios is None:
        st.warning("monte_carlo_scenarios.csv not found in repository root.")
    else:
        c1,c2,c3,c4 = st.columns(4)
        cac = scenarios["blended_cac_aud"]
        c1.metric("Median CAC",  f"AUD ${cac.median():.2f}")
        c2.metric("Mean CAC",    f"AUD ${cac.mean():.2f}")
        c3.metric("5th pct",     f"AUD ${cac.quantile(0.05):.2f}")
        c4.metric("95th pct",    f"AUD ${cac.quantile(0.95):.2f}")
        st.divider()

        col_l,col_r = st.columns(2)
        with col_l:
            fig4 = px.histogram(scenarios,x="blended_cac_aud",nbins=60,
                                title="CAC Distribution — 9,999 scenarios",
                                color_discrete_sequence=["#2E6E62"])
            fig4.add_vline(x=cac.median(),line_dash="dash",
                           line_color="#E8A838",annotation_text="Median")
            fig4.update_layout(height=360)
            st.plotly_chart(fig4,use_container_width=True)
        with col_r:
            s = scenarios.sample(min(2000,len(scenarios)),random_state=42)
            fig5 = px.scatter(s,x="google_spend",y="blended_cac_aud",
                              color="meta_spend",color_continuous_scale="Blues",
                              title="Google Spend vs Blended CAC",opacity=0.4,
                              height=360)
            st.plotly_chart(fig5,use_container_width=True)

        if model_eval is not None:
            st.subheader("Model Performance")
            st.dataframe(model_eval,use_container_width=True)

# ── TAB 4 — Sensitivity ───────────────────────────────────────────────────────
with tab4:
    st.subheader("Sensitivity Analysis")
    if sensitivity is None:
        st.warning("sensitivity_analysis.csv not found in repository root.")
    else:
        dims = sensitivity["Dimension"].unique().tolist()
        sel  = st.selectbox("Select dimension",dims)
        sub  = sensitivity[sensitivity["Dimension"]==sel]

        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=sub["Factor"],y=sub["Median CAC"],
                       mode="lines+markers",name="Median CAC",
                       line=dict(color="#1B3A6B",width=3),
                       marker=dict(size=8,color="#2E6E62")))
        fig6.add_trace(go.Scatter(x=sub["Factor"],y=sub["p95 CAC"],
                       mode="lines",name="95th pct",
                       line=dict(color="#E8A838",dash="dash")))
        fig6.add_trace(go.Scatter(x=sub["Factor"],y=sub["p5 CAC"],
                       mode="lines",name="5th pct",
                       line=dict(color="#E8A838",dash="dot"),
                       fill="tonexty",fillcolor="rgba(232,168,56,0.08)"))
        fig6.update_layout(title=f"Sensitivity: {sel}",
                           xaxis_title="Factor",yaxis_title="CAC (AUD)",
                           height=400)
        st.plotly_chart(fig6,use_container_width=True)
        st.dataframe(sensitivity,use_container_width=True,height=380)

st.divider()
st.caption("MDS650 Data Science Capstone · Sydney Polytechnic Institute · "
           "Bayesian Multi-Channel Attribution Framework")
