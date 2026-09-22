"""
CAC Compass — Channel Mix Optimiser
Professional marketing intelligence tool for Australian DTC brands.
Powered by Bayesian attribution modelling and Differential Evolution optimisation.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import differential_evolution
import os

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CAC Compass — Channel Mix Optimiser",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── BRAND COLOURS ─────────────────────────────────────────────────────────────
NAVY   = "#1A3260"
GOLD   = "#E1A03A"
TEAL   = "#1E7A6A"
GREY   = "#5A6474"
LGREY  = "#F4F6F9"
WHITE  = "#FFFFFF"
RED    = "#C0392B"
GREEN  = "#2E7A5A"

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ── fonts & base ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

  /* ── hide default streamlit chrome ── */
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding-top: 1rem; padding-bottom: 2rem; max-width: 1280px; }}

  /* ── sidebar ── */
  [data-testid="stSidebar"] {{
    background-color: {NAVY};
    padding-top: 1.5rem;
  }}
  [data-testid="stSidebar"] * {{ color: #C5D8E8 !important; }}
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {{
    color: {GOLD} !important;
    font-weight: 600;
  }}
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stNumberInput label {{
    color: #8AB4D0 !important;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    text-transform: uppercase;
  }}
  [data-testid="stSidebar"] hr {{
    border-color: #2E4A6A;
    margin: 1.2rem 0;
  }}

  /* ── metric cards ── */
  [data-testid="metric-container"] {{
    background: {WHITE};
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  [data-testid="metric-container"] label {{
    color: {GREY} !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
  }}
  [data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {NAVY} !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
  }}
  [data-testid="metric-container"] [data-testid="stMetricDelta"] {{
    font-size: 0.82rem !important;
  }}

  /* ── tab bar ── */
  [data-baseweb="tab-list"] {{
    gap: 0;
    background: {LGREY};
    border-radius: 10px;
    padding: 4px;
  }}
  [data-baseweb="tab"] {{
    border-radius: 8px !important;
    padding: 0.5rem 1.4rem !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    color: {GREY} !important;
  }}
  [aria-selected="true"][data-baseweb="tab"] {{
    background: {NAVY} !important;
    color: {WHITE} !important;
  }}

  /* ── buttons ── */
  .stButton > button {{
    background: {NAVY};
    color: {WHITE};
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.02em;
    transition: all 0.2s;
    width: 100%;
  }}
  .stButton > button:hover {{
    background: {TEAL};
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  }}

  /* ── divider ── */
  hr {{ border-color: #E2E8F0; margin: 1.5rem 0; }}

  /* ── cards ── */
  .card {{
    background: {WHITE};
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
  }}
  .card-navy {{
    background: {NAVY};
    border-radius: 12px;
    padding: 1.5rem;
    color: {WHITE};
  }}
  .kpi-big {{
    font-size: 2.4rem;
    font-weight: 700;
    color: {NAVY};
    line-height: 1;
  }}
  .kpi-label {{
    font-size: 0.78rem;
    font-weight: 600;
    color: {GREY};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.3rem;
  }}
  .kpi-sub {{
    font-size: 0.82rem;
    color: {GREY};
    margin-top: 0.2rem;
  }}
  .tag {{
    display: inline-block;
    background: {LGREY};
    color: {NAVY};
    border-radius: 6px;
    padding: 0.2rem 0.7rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.3rem;
  }}
  .tag-gold {{
    background: #FDF3E0;
    color: #8A5A0A;
  }}
  .tag-green {{
    background: #E0F2EC;
    color: #1A5C3A;
  }}
  .tag-red {{
    background: #FDEDEC;
    color: #8A1A1A;
  }}
  .section-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {NAVY};
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid {GOLD};
    display: inline-block;
  }}
  .insight-box {{
    background: linear-gradient(135deg, {NAVY} 0%, #2C4A7A 100%);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    color: {WHITE};
    margin: 0.5rem 0;
  }}
  .insight-box .label {{
    font-size: 0.72rem;
    color: #8AB4D0;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }}
  .insight-box .value {{
    font-size: 1.8rem;
    font-weight: 700;
    color: {GOLD};
    line-height: 1.2;
  }}
  .insight-box .sub {{
    font-size: 0.82rem;
    color: #C5D8E8;
    margin-top: 0.2rem;
  }}
  .warning-box {{
    background: #FFF8E8;
    border-left: 4px solid {GOLD};
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    font-size: 0.85rem;
    color: #5A3A0A;
    margin: 0.8rem 0;
  }}
  .info-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid #F0F2F5;
    font-size: 0.88rem;
  }}
  .info-row:last-child {{ border-bottom: none; }}
  .info-row .key {{ color: {GREY}; }}
  .info-row .val {{ color: {NAVY}; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
CHANNELS     = ["Meta", "Google Ads", "TikTok"]
FEATURE_COLS = ["meta_spend","google_spend","tiktok_spend",
                "seasonality_index","adstock","saturation"]
CH_COLORS    = {"Meta": NAVY, "Google Ads": TEAL, "TikTok": "#7B4FA0"}
CH_ICONS     = {"Meta": "📘", "Google Ads": "🔍", "TikTok": "🎵"}

def feat(m, g, t, seas=1.0, ads=1.0, sat=1.0):
    return pd.DataFrame([[m, g, t, seas, ads, sat]], columns=FEATURE_COLS)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("best_cac_model.joblib")

@st.cache_data(show_spinner=False)
def load_csv(path):
    try: return pd.read_csv(path)
    except: return None

@st.cache_data(show_spinner=False)
def load_priors():
    try:
        return pd.read_excel(
            "MDS650_Bayesian_Priors_READY_FOR_PYTHON.xlsx",
            sheet_name="PYTHON_Priors"
        )
    except:
        return None

with st.spinner(""):
    model      = load_model()
    scenarios  = load_csv("monte_carlo_scenarios.csv")
    sensitivity= load_csv("sensitivity_analysis.csv")
    model_eval = load_csv("model_evaluation.csv")
    priors     = load_priors()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding-bottom:1.2rem; border-bottom:1px solid #2E4A6A;'>
      <div style='font-size:1.6rem; font-weight:800; color:{GOLD}; letter-spacing:-0.02em;'>
        🧭 CAC Compass
      </div>
      <div style='font-size:0.76rem; color:#8AB4D0; margin-top:0.2rem; letter-spacing:0.04em;'>
        CHANNEL MIX OPTIMISER
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Weekly Budget")
    budget = st.number_input(
        "Total spend (AUD $)",
        min_value=1_000, max_value=200_000,
        value=10_000, step=500,
        help="Your total weekly digital advertising budget across all channels."
    )

    st.markdown("---")
    st.markdown("### Manual Split")
    st.caption("Explore a custom allocation below.")
    meta_p   = st.slider("Meta (%)",   5, 90, 6,  key="meta_pct")
    google_p = st.slider("Google Ads (%)", 5, 90, 89, key="google_pct")
    tiktok_p = max(5, min(90, 100 - meta_p - google_p))
    st.markdown(f"""
    <div style='background:#0F2744; border-radius:8px; padding:0.6rem 0.9rem; margin-top:0.3rem;'>
      <span style='color:#8AB4D0; font-size:0.78rem; font-weight:600;'>TikTok</span><br/>
      <span style='color:{GOLD}; font-size:1.2rem; font-weight:700;'>{tiktok_p}%</span>
      <span style='color:#8AB4D0; font-size:0.78rem;'> auto-calculated</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Assumptions")
    ltv = st.number_input("Customer LTV (AUD $)", min_value=50, max_value=2000, value=400, step=50)
    with st.expander("Advanced controls"):
        seas = st.slider("Seasonality index", 0.80, 1.20, 1.00, 0.01)
        ads_mult = st.slider("Adstock multiplier", 0.85, 1.15, 1.00, 0.01)
        sat  = st.slider("Saturation multiplier", 0.75, 1.05, 1.00, 0.01)

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.72rem; color:#4A6A8A; line-height:1.6;'>
      Powered by Random Forest + Differential Evolution.<br/>
      Trained on 9,999 Bayesian simulation scenarios.<br/>
      Australian DTC health supplement benchmarks.
    </div>
    """, unsafe_allow_html=True)

# ── MAIN HEADER ───────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown(f"""
    <h1 style='font-size:1.9rem; font-weight:800; color:{NAVY};
               margin:0; letter-spacing:-0.02em;'>
      Channel Mix Optimiser
    </h1>
    <p style='color:{GREY}; font-size:0.9rem; margin:0.3rem 0 0;'>
      AI-powered budget allocation for Australian DTC health supplement brands
    </p>
    """, unsafe_allow_html=True)
with col_badge:
    st.markdown(f"""
    <div style='text-align:right; padding-top:0.4rem;'>
      <span class='tag'>🇦🇺 AU Market</span>
      <span class='tag tag-green'>Live</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  🎯  Smart Allocation  ",
    "  🔬  Scenario Explorer  ",
    "  📐  Manual Analysis  ",
    "  🌡️  Sensitivity  ",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SMART ALLOCATION (DE OPTIMISER)
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<br/>", unsafe_allow_html=True)

    run_col, _ = st.columns([1, 2])
    with run_col:
        run = st.button("🚀  Run Smart Allocation", use_container_width=True)

    if run:
        with st.spinner("Optimising allocation across 60 candidate solutions…"):
            def obj(x):
                s = np.clip(x, 0.05, 0.80); s = s / s.sum()
                return model.predict(feat(s[0]*budget, s[1]*budget, s[2]*budget,
                                          seas, ads_mult, sat))[0]
            res = differential_evolution(obj, [(0.05, 0.80)]*3,
                  seed=42, maxiter=500, tol=1e-10, popsize=20, workers=1)
            opt = np.clip(res.x, 0.05, 0.80); opt = opt / opt.sum()

        pred_cac = res.fun
        eq_cac   = model.predict(feat(budget/3, budget/3, budget/3,
                                      seas, ads_mult, sat))[0]
        improvement = (1 - pred_cac / eq_cac) * 100
        ratio = ltv / pred_cac if pred_cac > 0 else 0
        ratio_ok = ratio >= 3

        # ── Top KPIs ─────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Predicted CAC", f"AUD ${pred_cac:,.2f}")
        with k2:
            st.metric("LTV:CAC Ratio", f"{ratio:.1f}×",
                      delta="Above 3:1 ✓" if ratio_ok else "Below 3:1 ✗",
                      delta_color="normal" if ratio_ok else "inverse")
        with k3:
            st.metric("vs Equal Split", f"−{improvement:.1f}%",
                      delta=f"AUD ${(eq_cac - pred_cac):,.0f} saving per customer")
        with k4:
            wkly_customers = budget / pred_cac if pred_cac > 0 else 0
            st.metric("Est. Weekly Customers", f"{wkly_customers:.0f}")

        st.markdown("<br/>", unsafe_allow_html=True)

        # ── Allocation cards + donut ──────────────────────────────────────────
        left, right = st.columns([1, 1])

        with left:
            st.markdown(f"<div class='section-title'>Recommended Allocation</div>",
                        unsafe_allow_html=True)
            for ch, share in zip(CHANNELS, opt):
                spend = share * budget
                color = CH_COLORS[ch]
                icon  = CH_ICONS[ch]
                pct   = share * 100
                st.markdown(f"""
                <div style='background:{WHITE}; border:1px solid #E2E8F0;
                            border-left:4px solid {color}; border-radius:0 10px 10px 0;
                            padding:1rem 1.3rem; margin-bottom:0.7rem;
                            box-shadow:0 1px 3px rgba(0,0,0,0.05);'>
                  <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                      <div style='font-size:0.78rem; color:{GREY}; font-weight:600;
                                  text-transform:uppercase; letter-spacing:0.04em;'>
                        {icon} {ch}
                      </div>
                      <div style='font-size:1.6rem; font-weight:800; color:{color};
                                  line-height:1.2; margin-top:0.15rem;'>
                        {pct:.1f}%
                      </div>
                    </div>
                    <div style='text-align:right;'>
                      <div style='font-size:1.1rem; font-weight:700; color:{NAVY};'>
                        AUD ${spend:,.0f}
                      </div>
                      <div style='font-size:0.78rem; color:{GREY};'>per week</div>
                    </div>
                  </div>
                  <div style='margin-top:0.6rem; background:#F0F2F5;
                              border-radius:4px; height:6px;'>
                    <div style='background:{color}; height:6px; border-radius:4px;
                                width:{min(pct,100):.1f}%;'></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with right:
            # Donut chart
            fig_donut = go.Figure(go.Pie(
                labels=CHANNELS,
                values=[opt[i]*budget for i in range(3)],
                hole=0.62,
                marker_colors=[CH_COLORS[c] for c in CHANNELS],
                textinfo="label+percent",
                textfont_size=13,
                hovertemplate="<b>%{label}</b><br>AUD $%{value:,.0f}<br>%{percent}<extra></extra>",
            ))
            fig_donut.add_annotation(
                text=f"<b>AUD ${pred_cac:,.0f}</b><br><span style='font-size:11px'>Predicted CAC</span>",
                x=0.5, y=0.5, showarrow=False, font_size=16,
                font_color=NAVY, align="center"
            )
            fig_donut.update_layout(
                showlegend=False, height=320, margin=dict(t=20,b=20,l=20,r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # ── CAC surface plot ──────────────────────────────────────────────────
        st.markdown(f"<div class='section-title'>CAC Surface — Google Ads Concentration</div>",
                    unsafe_allow_html=True)
        st.caption("How predicted CAC changes as Google Ads share increases (remainder split equally between Meta and TikTok)")

        g_shares  = np.linspace(0.05, 0.80, 80)
        cac_curve = [model.predict(feat(
            (1-g)/2*budget, g*budget, (1-g)/2*budget, seas, ads_mult, sat
        ))[0] for g in g_shares]

        fig_surf = go.Figure()
        fig_surf.add_trace(go.Scatter(
            x=g_shares*100, y=cac_curve, mode="lines",
            line=dict(color=NAVY, width=2.5),
            fill="tozeroy", fillcolor="rgba(26,50,96,0.06)",
            hovertemplate="Google %{x:.1f}%<br>CAC: AUD $%{y:,.2f}<extra></extra>",
            name="Predicted CAC"
        ))
        fig_surf.add_vline(x=33.3, line_dash="dot", line_color=GREY, line_width=1.5,
            annotation_text="Equal split", annotation_font_color=GREY, annotation_font_size=11)
        fig_surf.add_vline(x=opt[1]*100, line_dash="dash", line_color=GOLD, line_width=2,
            annotation_text=f"Optimal {opt[1]*100:.0f}%",
            annotation_font_color=GOLD, annotation_font_size=11)
        fig_surf.update_layout(
            xaxis_title="Google Ads Share (%)", yaxis_title="Customer Acquisition Cost (AUD)",
            height=300, margin=dict(t=20,b=50,l=60,r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#F0F2F5", showgrid=True),
            yaxis=dict(gridcolor="#F0F2F5", showgrid=True),
            showlegend=False,
        )
        st.plotly_chart(fig_surf, use_container_width=True)

        # ── Warning ───────────────────────────────────────────────────────────
        st.markdown(f"""
        <div class='warning-box'>
          ⚠️ <strong>Important:</strong> Google Ads CVR benchmark (6.80%) is derived from a
          health & fitness category proxy. Validate with 4 weeks of live campaign data before
          committing to this allocation at scale.
        </div>
        """, unsafe_allow_html=True)

    else:
        # Landing state
        st.markdown(f"""
        <div style='background:{LGREY}; border-radius:16px; padding:3rem;
                    text-align:center; margin:1rem 0;'>
          <div style='font-size:2.8rem; margin-bottom:1rem;'>🧭</div>
          <h2 style='color:{NAVY}; font-size:1.4rem; font-weight:700; margin:0 0 0.8rem;'>
            Ready to optimise your channel mix
          </h2>
          <p style='color:{GREY}; font-size:0.92rem; max-width:480px;
                    margin:0 auto 1.5rem; line-height:1.6;'>
            Set your weekly budget in the sidebar, then click
            <strong>Run Smart Allocation</strong> to find the optimal
            Meta / Google Ads / TikTok split using Differential Evolution.
          </p>
          <div style='display:flex; justify-content:center; gap:2rem; flex-wrap:wrap;'>
            <div>
              <div style='font-size:1.5rem; font-weight:800; color:{NAVY};'>9,999</div>
              <div style='font-size:0.78rem; color:{GREY};'>Training scenarios</div>
            </div>
            <div>
              <div style='font-size:1.5rem; font-weight:800; color:{NAVY};'>R² 0.64</div>
              <div style='font-size:0.78rem; color:{GREY};'>Model accuracy</div>
            </div>
            <div>
              <div style='font-size:1.5rem; font-weight:800; color:{NAVY};'>60.5%</div>
              <div style='font-size:0.78rem; color:{GREY};'>Avg CAC reduction</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SCENARIO EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<br/>", unsafe_allow_html=True)

    if scenarios is None:
        st.warning("Scenario data not found. Upload `monte_carlo_scenarios.csv` to the repository root.")
        st.stop()

    cac = scenarios["blended_cac_aud"]

    # Top KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("Scenarios Run", f"{len(scenarios):,}")
    with k2: st.metric("Median CAC", f"AUD ${cac.median():,.2f}")
    with k3: st.metric("Mean CAC", f"AUD ${cac.mean():,.2f}")
    with k4: st.metric("Best 5%", f"AUD ${cac.quantile(0.05):,.2f}")
    with k5: st.metric("Worst 5%", f"AUD ${cac.quantile(0.95):,.2f}")

    st.markdown("<br/>", unsafe_allow_html=True)

    l_col, r_col = st.columns(2)

    with l_col:
        st.markdown(f"<div class='section-title'>CAC Distribution</div>",
                    unsafe_allow_html=True)
        st.caption("Across 9,999 random budget allocation scenarios (AUD $10,000 total budget)")

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=cac.clip(upper=cac.quantile(0.99)),
            nbinsx=60,
            marker_color=NAVY, marker_opacity=0.85,
            hovertemplate="CAC: AUD $%{x:,.0f}<br>Count: %{y}<extra></extra>",
        ))
        fig_hist.add_vline(x=cac.median(), line_dash="dash",
            line_color=GOLD, line_width=2,
            annotation_text=f"Median AUD ${cac.median():,.0f}",
            annotation_font_color=GOLD, annotation_font_size=11)
        fig_hist.update_layout(
            xaxis_title="Customer Acquisition Cost (AUD)",
            yaxis_title="Number of Scenarios",
            height=340, margin=dict(t=10,b=50,l=60,r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#F0F2F5"), yaxis=dict(gridcolor="#F0F2F5"),
            showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with r_col:
        st.markdown(f"<div class='section-title'>Google Ads Spend vs CAC</div>",
                    unsafe_allow_html=True)
        st.caption("Each point is one simulated scenario — higher Google spend consistently reduces CAC")

        sample = scenarios.sample(min(3000, len(scenarios)), random_state=42)
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=sample["google_spend"],
            y=sample["blended_cac_aud"].clip(upper=sample["blended_cac_aud"].quantile(0.98)),
            mode="markers",
            marker=dict(
                color=sample["meta_spend"],
                colorscale=[[0, "#E8F0F8"], [1, NAVY]],
                size=5, opacity=0.45,
                colorbar=dict(title="Meta<br>Spend", thickness=12, len=0.7),
            ),
            hovertemplate="Google: AUD $%{x:,.0f}<br>CAC: AUD $%{y:,.2f}<extra></extra>",
        ))
        fig_sc.update_layout(
            xaxis_title="Google Ads Spend (AUD)",
            yaxis_title="Customer Acquisition Cost (AUD)",
            height=340, margin=dict(t=10,b=50,l=60,r=60),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#F0F2F5"), yaxis=dict(gridcolor="#F0F2F5"),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # Model performance
    if model_eval is not None:
        st.markdown(f"<div class='section-title'>Model Performance</div>",
                    unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            st.dataframe(
                model_eval.style.format({
                    "RMSE":"AUD ${:.2f}", "MAE":"AUD ${:.2f}", "R2":"{:.3f}"
                }).set_properties(**{"background-color": WHITE}),
                use_container_width=True, hide_index=True
            )
        with m2:
            st.markdown(f"""
            <div class='card'>
              <div style='font-size:0.82rem; color:{GREY}; line-height:1.7;'>
                <strong style='color:{NAVY};'>Random Forest</strong> was selected over Bayesian Ridge
                due to 26.2% lower test RMSE and superior 5-fold cross-validation stability.
                <br/><br/>
                The model explains <strong>64.4%</strong> of CAC variance across 9,999 simulated
                scenarios with <strong>google_spend</strong> dominating feature importance at
                <strong>74.7%</strong>.
              </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MANUAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<br/>", unsafe_allow_html=True)

    shares = np.array([meta_p, google_p, tiktok_p]) / 100
    spends = shares * budget
    pred   = model.predict(feat(*spends, seas, ads_mult, sat))[0]
    ratio_manual = ltv / pred if pred > 0 else 0
    opt_pred = model.predict(feat(0.056*budget, 0.888*budget, 0.056*budget,
                                   seas, ads_mult, sat))[0]
    eq_pred  = model.predict(feat(budget/3, budget/3, budget/3,
                                   seas, ads_mult, sat))[0]
    viable = ratio_manual >= 3

    # Allocation summary
    a1, a2, a3 = st.columns(3)
    for col, ch, sp, sh in zip([a1,a2,a3], CHANNELS, spends, shares):
        with col:
            color = CH_COLORS[ch]
            icon  = CH_ICONS[ch]
            st.markdown(f"""
            <div style='background:{WHITE}; border:1px solid #E2E8F0;
                        border-top:4px solid {color}; border-radius:10px;
                        padding:1.2rem; text-align:center;
                        box-shadow:0 1px 4px rgba(0,0,0,0.05);'>
              <div style='font-size:1.1rem;'>{icon}</div>
              <div style='font-size:0.78rem; color:{GREY}; font-weight:600;
                          text-transform:uppercase; letter-spacing:0.05em;'>{ch}</div>
              <div style='font-size:1.7rem; font-weight:800; color:{color};
                          line-height:1.2; margin:0.2rem 0;'>{sh*100:.0f}%</div>
              <div style='font-size:0.9rem; color:{NAVY}; font-weight:600;'>
                AUD ${sp:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown(f"<div class='section-title'>Performance vs Benchmarks</div>",
                    unsafe_allow_html=True)

        fig_bar = go.Figure()
        labels   = ["Your Allocation", "Smart Allocation", "Equal Split"]
        values   = [pred, opt_pred, eq_pred]
        colors   = [TEAL, GOLD, GREY]

        fig_bar.add_trace(go.Bar(
            x=labels, y=values,
            marker_color=colors,
            text=[f"AUD ${v:,.0f}" for v in values],
            textposition="outside",
            textfont=dict(size=13, color=NAVY),
            hovertemplate="<b>%{x}</b><br>CAC: AUD $%{y:,.2f}<extra></extra>",
        ))
        fig_bar.update_layout(
            yaxis_title="Customer Acquisition Cost (AUD)",
            yaxis_range=[0, max(values)*1.3],
            height=320, margin=dict(t=30,b=40,l=60,r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#F0F2F5"),
            yaxis=dict(gridcolor="#F0F2F5"),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with right:
        st.markdown(f"<div class='section-title'>Business Health Check</div>",
                    unsafe_allow_html=True)

        st.markdown(f"""
        <div class='insight-box'>
          <div class='label'>Predicted Customer Acquisition Cost</div>
          <div class='value'>AUD ${pred:,.2f}</div>
          <div class='sub'>per new customer acquired</div>
        </div>
        <div class='insight-box' style='margin-top:0.8rem;'>
          <div class='label'>LTV:CAC Ratio (at AUD ${ltv:,} LTV)</div>
          <div class='value' style='color:{"#4CAF50" if viable else "#FF6B6B"};'>
            {ratio_manual:.1f}×
          </div>
          <div class='sub'>{"✓  Above the 3:1 sustainability threshold" if viable
                           else "✗  Below the 3:1 sustainability threshold"}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        weekly_customers = budget / pred if pred > 0 else 0
        monthly_rev = weekly_customers * 4.33 * ltv

        for label, value in [
            ("Weekly customers", f"{weekly_customers:.0f}"),
            ("Monthly customer revenue", f"AUD ${monthly_rev:,.0f}"),
            ("Weekly ad spend", f"AUD ${budget:,}"),
            ("Monthly ad spend", f"AUD ${budget*4.33:,.0f}"),
            ("Saving vs equal split", f"AUD ${(eq_pred - pred)*weekly_customers:,.0f}/wk"),
        ]:
            st.markdown(f"""
            <div class='info-row'>
              <span class='key'>{label}</span>
              <span class='val'>{value}</span>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<br/>", unsafe_allow_html=True)

    if sensitivity is None:
        st.warning("Sensitivity data not found. Upload `sensitivity_analysis.csv` to the repository root.")
    else:
        dims = sensitivity["Dimension"].unique().tolist()

        l_col, r_col = st.columns([1, 2])

        with l_col:
            st.markdown(f"<div class='section-title'>Select Dimension</div>",
                        unsafe_allow_html=True)
            selected = st.radio(
                "", dims, label_visibility="collapsed",
                help="Select a sensitivity dimension to explore."
            )

            st.markdown("<br/>", unsafe_allow_html=True)

            # Sensitivity heatmap summary
            summary_data = []
            for dim in dims:
                sub = sensitivity[sensitivity["Dimension"] == dim]
                rng = sub["Median CAC"].max() - sub["Median CAC"].min()
                level = "HIGH" if rng > 30 else "MODERATE" if rng > 10 else ("ZERO" if rng < 1 else "LOW")
                summary_data.append({"Dimension": dim.split(". ",1)[-1], "Range": f"${rng:,.0f}", "Level": level})

            level_color = {"HIGH": RED, "MODERATE": GOLD, "LOW": GREEN, "ZERO": NAVY}
            for row in summary_data:
                col = level_color.get(row["Level"], GREY)
                st.markdown(f"""
                <div style='display:flex; align-items:center; padding:0.4rem 0;
                            border-bottom:1px solid #F0F2F5; font-size:0.82rem;'>
                  <span style='background:{col}; color:white; border-radius:4px;
                               padding:0.1rem 0.4rem; font-size:0.7rem; font-weight:700;
                               margin-right:0.6rem; min-width:4rem; text-align:center;'>
                    {row["Level"]}
                  </span>
                  <span style='color:{NAVY}; flex:1;'>{row["Dimension"]}</span>
                  <span style='color:{GREY}; font-weight:600;'>{row["Range"]}</span>
                </div>
                """, unsafe_allow_html=True)

        with r_col:
            st.markdown(f"<div class='section-title'>{selected}</div>",
                        unsafe_allow_html=True)

            sub = sensitivity[sensitivity["Dimension"] == selected].copy()

            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(
                x=sub["Factor"], y=sub["p95 CAC"], mode="lines",
                line=dict(color="#E2E8F0", width=0),
                fill=None, showlegend=False,
                hovertemplate="Factor %{x:.2f}<br>95th pct: AUD $%{y:,.0f}<extra></extra>",
            ))
            fig_sens.add_trace(go.Scatter(
                x=sub["Factor"], y=sub["p5 CAC"], mode="lines",
                line=dict(color="#E2E8F0", width=0),
                fill="tonexty", fillcolor="rgba(26,50,96,0.08)",
                name="5th–95th pct range",
                hovertemplate="Factor %{x:.2f}<br>5th pct: AUD $%{y:,.0f}<extra></extra>",
            ))
            fig_sens.add_trace(go.Scatter(
                x=sub["Factor"], y=sub["Median CAC"], mode="lines+markers",
                line=dict(color=NAVY, width=2.5),
                marker=dict(size=8, color=GOLD, line=dict(color=NAVY, width=2)),
                name="Median CAC",
                hovertemplate="Factor %{x:.2f}<br>Median CAC: AUD $%{y:,.2f}<extra></extra>",
            ))
            fig_sens.update_layout(
                xaxis_title="Factor Level",
                yaxis_title="Customer Acquisition Cost (AUD)",
                height=380, margin=dict(t=10,b=50,l=60,r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#F0F2F5"), yaxis=dict(gridcolor="#F0F2F5"),
                legend=dict(orientation="h", y=-0.15, font_size=12),
            )
            st.plotly_chart(fig_sens, use_container_width=True)

            # Data table
            display = sub[["Factor","Median CAC","Mean CAC","p5 CAC","p95 CAC"]].copy()
            display.columns = ["Factor","Median CAC","Mean CAC","5th pct","95th pct"]
            st.dataframe(
                display.style.format({
                    "Median CAC":"AUD ${:,.2f}", "Mean CAC":"AUD ${:,.2f}",
                    "5th pct":"AUD ${:,.2f}", "95th pct":"AUD ${:,.2f}",
                }).set_properties(**{"background-color": WHITE, "font-size": "13px"}),
                use_container_width=True, hide_index=True
            )

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown(f"""
<div style='border-top:1px solid #E2E8F0; padding-top:1.2rem;
            display:flex; justify-content:space-between; align-items:center;
            flex-wrap:wrap; gap:0.5rem;'>
  <div style='font-size:0.8rem; color:{GREY};'>
    <strong style='color:{NAVY};'>CAC Compass</strong> —
    Bayesian Multi-Channel Attribution ·
    Australian DTC Health Supplement Market ·
    Differential Evolution Optimisation
  </div>
  <div style='font-size:0.78rem; color:#A0AAB4;'>
    Model: Random Forest · R² 0.644 · 9,999 simulation scenarios ·
    FX: 1 USD = 1.5506 AUD (ATO 2025)
  </div>
</div>
""", unsafe_allow_html=True)
