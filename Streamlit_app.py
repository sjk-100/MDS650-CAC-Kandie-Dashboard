"""
CAC Compass — Channel Mix Optimiser
Bayesian simulation · Random Forest prediction · Differential Evolution optimisation
Australian DTC health supplement market
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import differential_evolution
import os

st.set_page_config(
    page_title="CAC Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY="#1A3260"; GOLD="#E1A03A"; TEAL="#1E7A6A"; GREY="#5A6474"
LGREY="#F4F6F9"; WHITE="#FFFFFF"; RED="#C0392B"; GREEN="#2E7A5A"

SENS_NOTES = {
    "1. Funnel (LPV x ATC)":
        "HIGH sensitivity. A 20% drop in landing-page or add-to-cart rates increases median CAC by ~57%. "
        "Improving on-site conversion delivers more value than channel reallocation.",
    "2. CVR prior mean":
        "HIGH sensitivity. A 20% reduction in conversion rate increases median CAC by ~87%. "
        "The Google Ads CVR assumption (6.80%) is your highest-risk input — validate with live data.",
    "3. CPM (all channels)":
        "MODERATE sensitivity. CAC scales roughly linearly with CPM. "
        "Negotiating lower CPM or shifting toward lower-CPM inventory reduces CAC proportionally.",
    "4. CTR prior mean":
        "MODERATE sensitivity. CTR affects the impressions-to-clicks stage non-linearly. "
        "Creative quality improvements on Google Ads have an outsized effect.",
    "5. Total budget":
        "MODERATE sensitivity. The Google-dominant allocation holds at all budget levels tested "
        "(AUD $5k–$20k). CAC improves slightly at higher spend due to Google efficiency at scale.",
    "6. Saturation mean":
        "LOW–MODERATE sensitivity. The 89% Google concentration faces mild diminishing returns. "
        "At very high budgets, spreading more to TikTok may reduce saturation risk.",
    "7. Seasonality std":
        "LOW sensitivity. Wider seasonality variance increases CAC spread but does not shift "
        "the median meaningfully. Seasonal risk is manageable.",
    "8. Adstock mean":
        "LOW sensitivity. Carryover effects from prior advertising are present but mild.",
    "9. Prior conc. (ESS)":
        "LOW sensitivity. Results are robust to prior strength — the framework is not "
        "overly sensitive to how tightly the benchmarks are trusted.",
    "10. CPC sigma_log":
        "ZERO sensitivity. CPC uncertainty has no effect on CAC. The model uses a "
        "CPM-based impression pathway, so CPC benchmarks do not enter the formula.",
    "10. CPC sigma_log (uncertainty)":
        "ZERO sensitivity. CPC uncertainty has no effect on CAC. The model uses a "
        "CPM-based impression pathway, so CPC benchmarks do not enter the formula.",
}

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html,body,[class*="css"]{{font-family:'Inter',sans-serif;}}
  #MainMenu,footer,header{{visibility:hidden;}}
  .block-container{{padding-top:1rem;padding-bottom:2rem;max-width:1280px;}}
  [data-testid="stSidebar"]{{background-color:{NAVY};padding-top:1.5rem;}}
  [data-testid="stSidebar"] *{{color:#C5D8E8!important;}}
  [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{{color:{GOLD}!important;font-weight:600;}}
  [data-testid="stSidebar"] label{{color:#8AB4D0!important;font-size:0.82rem;font-weight:500;letter-spacing:0.03em;text-transform:uppercase;}}
  [data-testid="stSidebar"] hr{{border-color:#2E4A6A;margin:1.2rem 0;}}
  [data-testid="metric-container"]{{background:{WHITE};border:1px solid #E2E8F0;border-radius:12px;padding:1.2rem 1.4rem;box-shadow:0 1px 4px rgba(0,0,0,0.06);}}
  [data-testid="metric-container"] label{{color:{GREY}!important;font-size:0.78rem!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:0.06em!important;}}
  [data-testid="metric-container"] [data-testid="stMetricValue"]{{color:{NAVY}!important;font-size:1.75rem!important;font-weight:700!important;}}
  [data-baseweb="tab-list"]{{gap:0;background:{LGREY};border-radius:10px;padding:4px;}}
  [data-baseweb="tab"]{{border-radius:8px!important;padding:0.5rem 1.4rem!important;font-weight:500!important;font-size:0.88rem!important;color:{GREY}!important;}}
  [aria-selected="true"][data-baseweb="tab"]{{background:{NAVY}!important;color:{WHITE}!important;}}
  .stButton>button{{background:{NAVY};color:{WHITE};border:none;border-radius:8px;padding:0.6rem 2rem;font-weight:600;font-size:0.9rem;letter-spacing:0.02em;transition:all 0.2s;width:100%;}}
  .stButton>button:hover{{background:{TEAL};transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,0,0,0.15);}}
  .section-title{{font-size:1.0rem;font-weight:700;color:{NAVY};margin-bottom:0.8rem;padding-bottom:0.5rem;border-bottom:2px solid {GOLD};display:inline-block;}}
  .card{{background:{WHITE};border:1px solid #E2E8F0;border-radius:12px;padding:1.5rem;box-shadow:0 1px 4px rgba(0,0,0,0.05);margin-bottom:1rem;}}
  .insight-box{{background:linear-gradient(135deg,{NAVY} 0%,#2C4A7A 100%);border-radius:10px;padding:1.2rem 1.5rem;color:{WHITE};margin:0.5rem 0;}}
  .info-row{{display:flex;justify-content:space-between;align-items:center;padding:0.6rem 0;border-bottom:1px solid #F0F2F5;font-size:0.88rem;}}
  .info-row:last-child{{border-bottom:none;}}
  .sens-note{{background:#F0F8FF;border-left:4px solid {TEAL};border-radius:0 8px 8px 0;padding:0.9rem 1.2rem;font-size:0.86rem;color:#1A3A5A;line-height:1.6;margin:0.8rem 0;}}
  hr{{border-color:#E2E8F0;margin:1.5rem 0;}}
</style>
""", unsafe_allow_html=True)

CHANNELS=["Meta","Google Ads","TikTok"]
FEATURE_COLS=["meta_spend","google_spend","tiktok_spend","seasonality_index","adstock","saturation"]
CH_COLORS={"Meta":NAVY,"Google Ads":TEAL,"TikTok":"#7B4FA0"}
CH_ICONS={"Meta":"📘","Google Ads":"🔍","TikTok":"🎵"}

def feat(m,g,t,seas=1.0,ads=1.0,sat=1.0):
    return pd.DataFrame([[m,g,t,seas,ads,sat]],columns=FEATURE_COLS)

def run_de(budget,seas,ads,sat):
    def obj(x):
        s=np.clip(x,0.05,0.80);s=s/s.sum()
        return model.predict(feat(s[0]*budget,s[1]*budget,s[2]*budget,seas,ads,sat))[0]
    res=differential_evolution(obj,[(0.05,0.80)]*3,seed=42,maxiter=500,tol=1e-10,popsize=20,workers=1)
    opt=np.clip(res.x,0.05,0.80);opt=opt/opt.sum()
    return opt,res.fun

@st.cache_resource(show_spinner=False)
def load_model():
    try: return joblib.load("best_cac_model.joblib")
    except FileNotFoundError: return None

@st.cache_data(show_spinner=False)
def load_csv(path):
    try: return pd.read_csv(path)
    except: return None

@st.cache_data(show_spinner=False)
def load_priors():
    try: return pd.read_excel("MDS650_Bayesian_Priors_READY_FOR_PYTHON.xlsx",sheet_name="PYTHON_Priors")
    except: return None

model=load_model()
scenarios=load_csv("monte_carlo_scenarios.csv")
sensitivity=load_csv("sensitivity_analysis.csv")
model_eval=load_csv("model_evaluation.csv")

if model is None:
    st.error("**Model file not found.** Please ensure `best_cac_model.joblib` is in the repository root.")
    st.info("Run the Jupyter notebook to generate the model, then upload `best_cac_model.joblib` to GitHub.")
    st.stop()

n_scenarios=len(scenarios) if scenarios is not None else 9999
r2_val=0.644
if model_eval is not None and "R2" in model_eval.columns:
    try:
        rf=model_eval[model_eval.iloc[:,0].str.contains("Forest",na=False)]
        if not rf.empty: r2_val=float(rf["R2"].values[0])
    except: pass

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding-bottom:1.2rem;border-bottom:1px solid #2E4A6A;'>
      <div style='font-size:1.6rem;font-weight:800;color:{GOLD};letter-spacing:-0.02em;'>🧭 CAC Compass</div>
      <div style='font-size:0.76rem;color:#8AB4D0;margin-top:0.2rem;letter-spacing:0.04em;'>CHANNEL MIX OPTIMISER</div>
    </div>""",unsafe_allow_html=True)
    st.markdown("### Weekly Budget")
    budget=st.number_input("Total spend (AUD $)",min_value=1000,max_value=200000,value=10000,step=500)
    ltv=st.number_input("Customer lifetime value (AUD $)",min_value=50,max_value=5000,value=400,step=50,
                        help="Used to calculate your LTV:CAC ratio. Sustainability benchmark: 3:1.")
    st.markdown("---")
    st.markdown("### Manual Split")
    st.caption("Explore a custom allocation below.")
    meta_p=st.slider("Meta (%)",5,90,6)
    google_p=st.slider("Google Ads (%)",5,90,89)
    tiktok_p=max(5,min(90,100-meta_p-google_p))
    st.markdown(f"""<div style='background:#0F2744;border-radius:8px;padding:0.6rem 0.9rem;margin-top:0.3rem;'>
      <span style='color:#8AB4D0;font-size:0.78rem;font-weight:600;'>TikTok</span><br/>
      <span style='color:{GOLD};font-size:1.2rem;font-weight:700;'>{tiktok_p}%</span>
      <span style='color:#8AB4D0;font-size:0.78rem;'> auto-calculated</span></div>""",unsafe_allow_html=True)
    st.markdown("---")
    with st.expander("⚙️ Advanced model controls"):
        seas=st.slider("Seasonal demand index",0.80,1.20,1.00,0.01,
            help="1.0 = average week. Increase for peak season (e.g. Jan health resolutions).")
        ads_mult=st.slider("Carryover effect from past ads",0.85,1.15,1.00,0.01,
            help="How much residual impact prior advertising has on current conversions.")
        sat=st.slider("Diminishing returns factor",0.75,1.05,1.00,0.01,
            help="Reduce if you're near saturation on a channel. 1.0 = no diminishing returns.")
    if seas==1.0 and ads_mult==1.0 and sat==1.0:
        seas=ads_mult=sat=1.0
    else:
        st.caption("⚠️ Advanced controls active")
    st.markdown("---")
    st.markdown(f"""<div style='font-size:0.72rem;color:#4A6A8A;line-height:1.7;'>
      Random Forest · Differential Evolution<br/>{n_scenarios:,} Bayesian simulation scenarios<br/>
      Australian DTC health supplement market<br/>FX: 1 USD = 1.5506 AUD (ATO 2025)</div>""",unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
col_t,col_b=st.columns([3,1])
with col_t:
    st.markdown(f"""<h1 style='font-size:1.9rem;font-weight:800;color:{NAVY};margin:0;letter-spacing:-0.02em;'>
      Channel Mix Optimiser</h1>
      <p style='color:{GREY};font-size:0.9rem;margin:0.3rem 0 0;'>
      Bayesian simulation · Random Forest prediction · Differential Evolution optimisation · Australian DTC health supplement market</p>""",unsafe_allow_html=True)
with col_b:
    st.markdown(f"""<div style='text-align:right;padding-top:0.5rem;'>
      <span style='background:#E0F2EC;color:#1A5C3A;border-radius:6px;padding:0.2rem 0.7rem;font-size:0.78rem;font-weight:600;'>● Live</span>
      &nbsp;<span style='background:{LGREY};color:{NAVY};border-radius:6px;padding:0.2rem 0.7rem;font-size:0.78rem;font-weight:600;'>🇦🇺 AU Market</span></div>""",unsafe_allow_html=True)
st.markdown("<hr/>",unsafe_allow_html=True)

tab1,tab2,tab3,tab4=st.tabs(["🎯 Smart Allocation","🔬 Scenario Explorer","📐 Manual Analysis","🌡️ Sensitivity"])

# ═══ TAB 1 — SMART ALLOCATION ═════════════════════════════════════════════════
with tab1:
    st.markdown("<br/>",unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def default_result(budget,seas,ads,sat):
        return run_de(budget,seas,ads,sat)

    if "opt" not in st.session_state:
        with st.spinner("Computing optimal allocation…"):
            opt,pred_cac=default_result(budget,seas,ads_mult,sat)
        st.session_state.opt=opt; st.session_state.pred_cac=pred_cac

    run_col,dl_col,_=st.columns([1,1,2])
    with run_col:
        if st.button("🚀  Recalculate",use_container_width=True):
            with st.spinner("Optimising…"):
                opt,pred_cac=run_de(budget,seas,ads_mult,sat)
            st.session_state.opt=opt; st.session_state.pred_cac=pred_cac

    opt=st.session_state.opt; pred_cac=st.session_state.pred_cac
    eq_cac=model.predict(feat(budget/3,budget/3,budget/3,seas,ads_mult,sat))[0]
    imprv=(1-pred_cac/eq_cac)*100; ratio=ltv/pred_cac if pred_cac>0 else 0
    wkly=budget/pred_cac if pred_cac>0 else 0; ratio_ok=ratio>=3
    ltv_thr=ltv/3

    alloc_df=pd.DataFrame({"Channel":CHANNELS,"Share (%)":[round(s*100,1) for s in opt],
        "Budget (AUD $)":[round(s*budget,0) for s in opt],"Predicted CAC":pred_cac,
        "Weekly Budget":budget,"LTV":ltv,"LTV:CAC Ratio":round(ratio,2)})
    with dl_col:
        st.download_button("⬇️  Export Allocation",data=alloc_df.to_csv(index=False).encode(),
            file_name=f"cac_compass_aud{budget}.csv",mime="text/csv",use_container_width=True)

    st.markdown("<br/>",unsafe_allow_html=True)
    k1,k2,k3,k4=st.columns(4)
    with k1: st.metric("Predicted CAC",f"AUD ${pred_cac:,.2f}")
    with k2: st.metric("LTV:CAC Ratio",f"{ratio:.1f}×",
                        delta="Above 3:1 ✓" if ratio_ok else "Below 3:1 ✗",
                        delta_color="normal" if ratio_ok else "inverse")
    with k3: st.metric("vs Equal Split",f"−{imprv:.1f}%",
                        delta=f"AUD ${(eq_cac-pred_cac):,.0f} saving / customer")
    with k4: st.metric("Est. Weekly Customers",f"{wkly:.0f}")

    st.markdown("<br/>",unsafe_allow_html=True)
    left,right=st.columns(2)

    with left:
        st.markdown(f"<div class='section-title'>Recommended Allocation</div>",unsafe_allow_html=True)
        for ch,share in zip(CHANNELS,opt):
            spend=share*budget; color=CH_COLORS[ch]; icon=CH_ICONS[ch]; pct=share*100
            st.markdown(f"""<div style='background:{WHITE};border:1px solid #E2E8F0;border-left:4px solid {color};
              border-radius:0 10px 10px 0;padding:1rem 1.3rem;margin-bottom:0.7rem;box-shadow:0 1px 3px rgba(0,0,0,0.05);'>
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div><div style='font-size:0.78rem;color:{GREY};font-weight:600;text-transform:uppercase;letter-spacing:0.04em;'>
                  {icon} {ch}</div>
                  <div style='font-size:1.6rem;font-weight:800;color:{color};line-height:1.2;margin-top:0.15rem;'>{pct:.1f}%</div></div>
                <div style='text-align:right;'><div style='font-size:1.1rem;font-weight:700;color:{NAVY};'>AUD ${spend:,.0f}</div>
                  <div style='font-size:0.78rem;color:{GREY};'>per week</div></div></div>
              <div style='margin-top:0.6rem;background:#F0F2F5;border-radius:4px;height:6px;'>
                <div style='background:{color};height:6px;border-radius:4px;width:{min(pct,100):.1f}%;'></div></div>
            </div>""",unsafe_allow_html=True)

    with right:
        fig=go.Figure(go.Pie(labels=CHANNELS,values=[opt[i]*budget for i in range(3)],hole=0.62,
            marker_colors=[CH_COLORS[c] for c in CHANNELS],textinfo="label+percent",textfont_size=13,
            hovertemplate="<b>%{label}</b><br>AUD $%{value:,.0f}<br>%{percent}<extra></extra>"))
        fig.add_annotation(text=f"<b>AUD ${pred_cac:,.0f}</b><br><span style='font-size:11px'>Predicted CAC</span>",
            x=0.5,y=0.5,showarrow=False,font_size=16,font_color=NAVY,align="center")
        fig.update_layout(showlegend=False,height=320,margin=dict(t=20,b=20,l=20,r=20),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)

    st.markdown(f"<div class='section-title'>CAC Surface — Google Ads Concentration</div>",unsafe_allow_html=True)
    st.caption("The green line shows the maximum CAC for your LTV:CAC sustainability target. Adjust LTV in the sidebar to update it.")
    g_shares=np.linspace(0.05,0.80,80)
    cac_curve=[model.predict(feat((1-g)/2*budget,g*budget,(1-g)/2*budget,seas,ads_mult,sat))[0] for g in g_shares]
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=g_shares*100,y=cac_curve,mode="lines",line=dict(color=NAVY,width=2.5),
        fill="tozeroy",fillcolor="rgba(26,50,96,0.06)",
        hovertemplate="Google %{x:.1f}%<br>CAC: AUD $%{y:,.2f}<extra></extra>",name="Predicted CAC"))
    fig2.add_hline(y=ltv_thr,line_dash="dash",line_color=GREEN,line_width=1.5,
        annotation_text=f"Sustainability threshold  AUD ${ltv_thr:,.0f}",
        annotation_font_color=GREEN,annotation_font_size=11,annotation_position="top right")
    fig2.add_vline(x=33.3,line_dash="dot",line_color=GREY,line_width=1.2,
        annotation_text="Equal split",annotation_font_color=GREY,annotation_font_size=11)
    fig2.add_vline(x=opt[1]*100,line_dash="dash",line_color=GOLD,line_width=2,
        annotation_text=f"Optimal {opt[1]*100:.0f}%",annotation_font_color=GOLD,annotation_font_size=11)
    fig2.update_layout(xaxis_title="Google Ads Share (%)",yaxis_title="Customer Acquisition Cost (AUD)",
        height=300,margin=dict(t=30,b=50,l=60,r=20),paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",xaxis=dict(gridcolor="#F0F2F5"),yaxis=dict(gridcolor="#F0F2F5"),showlegend=False)
    st.plotly_chart(fig2,use_container_width=True)

    with st.expander("📋 Methodology note"):
        st.markdown(f"""The Google Ads CVR benchmark (6.80%) is derived from a health & fitness category proxy,
        not supplement-specific Australian data. Sensitivity analysis shows a 20% reduction in CVR increases
        predicted CAC by approximately 87%.

        **Recommendation:** run campaigns across all three channels for 4 weeks post-launch, then update prior
        assumptions with your own CTR, CVR, and CPC data before committing to this allocation at scale.

        Model: Random Forest Regressor · R² {r2_val:.3f} · {n_scenarios:,} training scenarios ·
        Optimiser: Differential Evolution (popsize=20, seed=42)""")

# ═══ TAB 2 — SCENARIO EXPLORER ════════════════════════════════════════════════
with tab2:
    st.markdown("<br/>",unsafe_allow_html=True)
    if scenarios is None:
        st.warning("Upload `monte_carlo_scenarios.csv` to the repository root to enable this view.")
    else:
        cac=scenarios["blended_cac_aud"]
        k1,k2,k3,k4,k5=st.columns(5)
        with k1: st.metric("Scenarios",f"{len(scenarios):,}")
        with k2: st.metric("Median CAC",f"AUD ${cac.median():,.2f}")
        with k3: st.metric("Mean CAC",f"AUD ${cac.mean():,.2f}")
        with k4: st.metric("Best 5%",f"AUD ${cac.quantile(0.05):,.2f}")
        with k5: st.metric("Worst 5%",f"AUD ${cac.quantile(0.95):,.2f}")
        st.markdown("<br/>",unsafe_allow_html=True)
        lc,rc=st.columns(2)
        with lc:
            st.markdown(f"<div class='section-title'>CAC Distribution</div>",unsafe_allow_html=True)
            st.caption(f"Across {len(scenarios):,} random allocation scenarios (AUD $10,000 total budget)")
            fh=go.Figure()
            fh.add_trace(go.Histogram(x=cac.clip(upper=cac.quantile(0.99)),nbinsx=60,
                marker_color=NAVY,marker_opacity=0.85,
                hovertemplate="CAC: AUD $%{x:,.0f}<br>Count: %{y}<extra></extra>"))
            fh.add_vline(x=cac.median(),line_dash="dash",line_color=GOLD,line_width=2,
                annotation_text=f"Median AUD ${cac.median():,.0f}",annotation_font_color=GOLD,annotation_font_size=11)
            fh.update_layout(xaxis_title="Customer Acquisition Cost (AUD)",yaxis_title="Number of Scenarios",
                height=340,margin=dict(t=10,b=50,l=60,r=20),paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",xaxis=dict(gridcolor="#F0F2F5"),yaxis=dict(gridcolor="#F0F2F5"),showlegend=False)
            st.plotly_chart(fh,use_container_width=True)
        with rc:
            st.markdown(f"<div class='section-title'>Google Ads Spend vs CAC</div>",unsafe_allow_html=True)
            st.caption("Higher Google concentration consistently reduces CAC — 1,000 representative scenarios shown")
            samp=scenarios.sample(min(1000,len(scenarios)),random_state=42)
            fs=go.Figure()
            fs.add_trace(go.Scatter(x=samp["google_spend"],
                y=samp["blended_cac_aud"].clip(upper=samp["blended_cac_aud"].quantile(0.98)),
                mode="markers",marker=dict(color=samp["meta_spend"],
                colorscale=[[0,"#E8F0F8"],[1,NAVY]],size=5,opacity=0.45,
                colorbar=dict(title="Meta Spend",thickness=12,len=0.7)),
                hovertemplate="Google: AUD $%{x:,.0f}<br>CAC: AUD $%{y:,.2f}<extra></extra>"))
            fs.update_layout(xaxis_title="Google Ads Spend (AUD)",yaxis_title="Customer Acquisition Cost (AUD)",
                height=340,margin=dict(t=10,b=50,l=60,r=60),paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",xaxis=dict(gridcolor="#F0F2F5"),yaxis=dict(gridcolor="#F0F2F5"))
            st.plotly_chart(fs,use_container_width=True)
        if model_eval is not None:
            st.markdown(f"<div class='section-title'>Model Performance</div>",unsafe_allow_html=True)
            m1,m2=st.columns(2)
            with m1:
                try: st.dataframe(model_eval.style.format({"RMSE":"AUD ${:.2f}","MAE":"AUD ${:.2f}","R2":"{:.3f}"}),use_container_width=True,hide_index=True)
                except: st.dataframe(model_eval,use_container_width=True,hide_index=True)
            with m2:
                st.markdown(f"""<div class='card'><div style='font-size:0.86rem;color:{GREY};line-height:1.7;'>
                <strong style='color:{NAVY};'>Random Forest</strong> selected over Bayesian Ridge —
                26.2% lower test RMSE and superior cross-validation stability.<br/><br/>
                <strong>google_spend</strong> dominates feature importance at <strong>74.7%</strong>,
                consistent with Google Ads generating 37× more conversions per dollar than Meta under AU benchmark priors.
                </div></div>""",unsafe_allow_html=True)

# ═══ TAB 3 — MANUAL ANALYSIS ══════════════════════════════════════════════════
with tab3:
    st.markdown("<br/>",unsafe_allow_html=True)
    shares=np.array([meta_p,google_p,tiktok_p])/100; spends=shares*budget
    pred=model.predict(feat(*spends,seas,ads_mult,sat))[0]
    opt_pred=model.predict(feat(0.056*budget,0.888*budget,0.056*budget,seas,ads_mult,sat))[0]
    eq_pred=model.predict(feat(budget/3,budget/3,budget/3,seas,ads_mult,sat))[0]
    ratio_m=ltv/pred if pred>0 else 0; viable=ratio_m>=3
    a1,a2,a3=st.columns(3)
    for col,ch,sp,sh in zip([a1,a2,a3],CHANNELS,spends,shares):
        with col:
            color=CH_COLORS[ch]; icon=CH_ICONS[ch]
            st.markdown(f"""<div style='background:{WHITE};border:1px solid #E2E8F0;border-top:4px solid {color};
              border-radius:10px;padding:1.2rem;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,0.05);'>
              <div style='font-size:1.1rem;'>{icon}</div>
              <div style='font-size:0.78rem;color:{GREY};font-weight:600;text-transform:uppercase;letter-spacing:0.05em;'>{ch}</div>
              <div style='font-size:1.7rem;font-weight:800;color:{color};line-height:1.2;margin:0.2rem 0;'>{sh*100:.0f}%</div>
              <div style='font-size:0.9rem;color:{NAVY};font-weight:600;'>AUD ${sp:,.0f}</div></div>""",unsafe_allow_html=True)
    st.markdown("<br/>",unsafe_allow_html=True)
    lc,rc=st.columns([1.2,1])
    with lc:
        st.markdown(f"<div class='section-title'>Performance vs Benchmarks</div>",unsafe_allow_html=True)
        fb=go.Figure()
        fb.add_trace(go.Bar(x=["Your Allocation","Smart Allocation","Equal Split"],y=[pred,opt_pred,eq_pred],
            marker_color=[TEAL,GOLD,GREY],text=[f"AUD ${v:,.0f}" for v in [pred,opt_pred,eq_pred]],
            textposition="outside",textfont=dict(size=13,color=NAVY),
            hovertemplate="<b>%{x}</b><br>CAC: AUD $%{y:,.2f}<extra></extra>"))
        fb.update_layout(yaxis_title="Customer Acquisition Cost (AUD)",yaxis_range=[0,max(pred,opt_pred,eq_pred)*1.3],
            height=320,margin=dict(t=30,b=40,l=60,r=20),paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",xaxis=dict(gridcolor="#F0F2F5"),yaxis=dict(gridcolor="#F0F2F5"),showlegend=False)
        st.plotly_chart(fb,use_container_width=True)
    with rc:
        st.markdown(f"<div class='section-title'>Business Health Check</div>",unsafe_allow_html=True)
        wkly_c=budget/pred if pred>0 else 0; monthly_rev=wkly_c*4.33*ltv
        st.markdown(f"""<div class='insight-box'><div style='font-size:0.72rem;color:#8AB4D0;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;'>Predicted Customer Acquisition Cost</div>
          <div style='font-size:1.8rem;font-weight:700;color:{GOLD};line-height:1.2;'>AUD ${pred:,.2f}</div>
          <div style='font-size:0.82rem;color:#C5D8E8;margin-top:0.2rem;'>per new customer acquired</div></div>
        <div class='insight-box' style='margin-top:0.8rem;'>
          <div style='font-size:0.72rem;color:#8AB4D0;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;'>LTV:CAC Ratio (at AUD ${ltv:,} LTV)</div>
          <div style='font-size:1.8rem;font-weight:700;color:{"#4CAF50" if viable else "#FF6B6B"};line-height:1.2;'>{ratio_m:.1f}×</div>
          <div style='font-size:0.82rem;color:#C5D8E8;margin-top:0.2rem;'>{"✓  Above the 3:1 sustainability threshold" if viable else "✗  Below 3:1 — reduce CAC or increase LTV"}</div>
        </div>""",unsafe_allow_html=True)
        st.markdown("<br/>",unsafe_allow_html=True)
        for label,value in [("Weekly customers acquired",f"{wkly_c:.0f}"),
                             ("Monthly customer revenue",f"AUD ${monthly_rev:,.0f}"),
                             ("Weekly ad spend",f"AUD ${budget:,}"),
                             ("CAC saving vs equal split",f"AUD ${(eq_pred-pred):.0f} / customer"),
                             ("Weekly saving vs equal split",f"AUD ${(eq_pred-pred)*wkly_c:,.0f}")]:
            st.markdown(f"""<div class='info-row'><span style='color:{GREY};'>{label}</span>
              <span style='color:{NAVY};font-weight:600;'>{value}</span></div>""",unsafe_allow_html=True)
        st.markdown("<br/>",unsafe_allow_html=True)
        manual_df=pd.DataFrame({"Channel":CHANNELS,"Share (%)":[meta_p,google_p,tiktok_p],
            "Budget (AUD $)":[round(s,0) for s in spends],"Predicted CAC":pred,"LTV:CAC Ratio":round(ratio_m,2)})
        st.download_button("⬇️  Export this analysis",data=manual_df.to_csv(index=False).encode(),
            file_name=f"cac_compass_manual_aud{budget}.csv",mime="text/csv")

# ═══ TAB 4 — SENSITIVITY ══════════════════════════════════════════════════════
with tab4:
    st.markdown("<br/>",unsafe_allow_html=True)
    if sensitivity is None:
        st.warning("Upload `sensitivity_analysis.csv` to the repository root to enable this view.")
    else:
        dims=sensitivity["Dimension"].unique().tolist()
        lc,rc=st.columns([1,2])
        with lc:
            st.markdown(f"<div class='section-title'>Dimensions</div>",unsafe_allow_html=True)
            selected=st.radio("",dims,label_visibility="collapsed")
            st.markdown("<br/>",unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.78rem;color:{GREY};font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.5rem;'>Sensitivity Summary</div>",unsafe_allow_html=True)
            lev_col={"HIGH":RED,"MODERATE":GOLD,"LOW":GREEN,"ZERO":NAVY}
            for dim in dims:
                sub_d=sensitivity[sensitivity["Dimension"]==dim]
                rng=sub_d["Median CAC"].max()-sub_d["Median CAC"].min()
                lev="HIGH" if rng>30 else "MODERATE" if rng>10 else "ZERO" if rng<1 else "LOW"
                col=lev_col.get(lev,GREY); label=dim.split(". ",1)[-1] if ". " in dim else dim
                st.markdown(f"""<div style='display:flex;align-items:center;padding:0.35rem 0;border-bottom:1px solid #F0F2F5;font-size:0.82rem;'>
                  <span style='background:{col};color:white;border-radius:4px;padding:0.1rem 0.4rem;font-size:0.68rem;font-weight:700;margin-right:0.6rem;min-width:3.8rem;text-align:center;'>{lev}</span>
                  <span style='color:{NAVY};flex:1;font-size:0.81rem;'>{label}</span>
                  <span style='color:{GREY};font-weight:600;font-size:0.81rem;'>AUD ${rng:,.0f}</span></div>""",unsafe_allow_html=True)
        with rc:
            sub=sensitivity[sensitivity["Dimension"]==selected].copy()
            st.markdown(f"<div class='section-title'>{selected}</div>",unsafe_allow_html=True)
            note=SENS_NOTES.get(selected)
            if not note:
                for k,v in SENS_NOTES.items():
                    if k.lower() in selected.lower() or selected.lower() in k.lower():
                        note=v; break
            if note:
                st.markdown(f"<div class='sens-note'>💡 {note}</div>",unsafe_allow_html=True)
            fs=go.Figure()
            fs.add_trace(go.Scatter(x=sub["Factor"],y=sub["p95 CAC"],mode="lines",
                line=dict(color="#E2E8F0",width=0),fill=None,showlegend=False,
                hovertemplate="Factor %{x:.2f}<br>95th pct: AUD $%{y:,.0f}<extra></extra>"))
            fs.add_trace(go.Scatter(x=sub["Factor"],y=sub["p5 CAC"],mode="lines",
                line=dict(color="#E2E8F0",width=0),fill="tonexty",
                fillcolor="rgba(26,50,96,0.08)",name="5th–95th pct range",
                hovertemplate="Factor %{x:.2f}<br>5th pct: AUD $%{y:,.0f}<extra></extra>"))
            fs.add_trace(go.Scatter(x=sub["Factor"],y=sub["Median CAC"],mode="lines+markers",
                line=dict(color=NAVY,width=2.5),marker=dict(size=8,color=GOLD,line=dict(color=NAVY,width=2)),
                name="Median CAC",hovertemplate="Factor %{x:.2f}<br>Median CAC: AUD $%{y:,.2f}<extra></extra>"))
            fs.add_hline(y=ltv/3,line_dash="dash",line_color=GREEN,line_width=1.5,
                annotation_text=f"Sustainability threshold  AUD ${ltv/3:,.0f}",
                annotation_font_color=GREEN,annotation_font_size=11,annotation_position="top right")
            fs.update_layout(xaxis_title="Factor Level",yaxis_title="Customer Acquisition Cost (AUD)",
                height=380,margin=dict(t=20,b=50,l=60,r=20),paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",xaxis=dict(gridcolor="#F0F2F5"),yaxis=dict(gridcolor="#F0F2F5"),
                legend=dict(orientation="h",y=-0.18,font_size=12))
            st.plotly_chart(fs,use_container_width=True)
            disp=sub[["Factor","Median CAC","Mean CAC","p5 CAC","p95 CAC"]].copy()
            disp.columns=["Factor","Median CAC","Mean CAC","5th pct","95th pct"]
            try: st.dataframe(disp.style.format({"Median CAC":"AUD ${:,.2f}","Mean CAC":"AUD ${:,.2f}","5th pct":"AUD ${:,.2f}","95th pct":"AUD ${:,.2f}"}),use_container_width=True,hide_index=True)
            except: st.dataframe(disp,use_container_width=True,hide_index=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<br/>",unsafe_allow_html=True)
st.markdown(f"""<div style='border-top:1px solid #E2E8F0;padding-top:1.2rem;display:flex;
  justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;'>
  <div style='font-size:0.8rem;color:{GREY};'>
    <strong style='color:{NAVY};'>CAC Compass</strong> —
    Bayesian simulation · Random Forest prediction · Differential Evolution · Australian DTC market
  </div>
  <div style='font-size:0.78rem;color:#A0AAB4;'>
    R² {r2_val:.3f} · {n_scenarios:,} scenarios · FX: 1 USD = 1.5506 AUD (ATO 2025)
  </div>
</div>""",unsafe_allow_html=True)
