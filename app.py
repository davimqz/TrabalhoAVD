import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pipeline import load_data, preprocess, build_features, train_models, get_pvalues, predict_price

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

BLUE  = "#58A6FF"
RED   = "#FF7B72"
AMBER = "#E3B341"
GREEN = "#3FB950"
DARK  = "#0D1117"

# ── Gestalt: Similaridade — função única de estilo para TODOS os gráficos ──────
def plotly_defaults(fig, show_legend=False, ygrid=True):
    """Aplica tema escuro consistente. Gestalt: Similaridade + Figura-Fundo."""
    fig.update_layout(
        plot_bgcolor="#1C2128", paper_bgcolor="#161B22",
        font=dict(family="Inter, sans-serif", color="#C9D1D9", size=12),
        margin=dict(t=24, b=24, l=10, r=24),
        showlegend=show_legend,
        legend=dict(bgcolor="rgba(22,27,34,.9)", bordercolor="#30363D",
                    borderwidth=1, font=dict(size=11)),
    )
    # Gestalt: Redução de Ruído — grid sutil, sem bordas de eixo, sem ticks
    fig.update_xaxes(
        showgrid=True, gridcolor="#21262D", gridwidth=0.5,
        zeroline=False, showline=False, ticks="",
        tickfont=dict(size=11, color="#8B949E"),
        title_font=dict(size=12, color="#8B949E"),
    )
    fig.update_yaxes(
        showgrid=ygrid, gridcolor="#21262D", gridwidth=0.5,
        zeroline=False, showline=False, ticks="",
        tickfont=dict(size=11, color="#8B949E"),
        title_font=dict(size=12, color="#8B949E"),
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# CSS — DARK THEME
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color-scheme: dark;
}

/* ── Fundo geral ──────────────────────────────────────────────────────────── */
.stApp, .stApp > div, section[data-testid="stSidebar"] {
    background: #0D1117 !important;
}
#MainMenu, footer { visibility: hidden; }
.main .block-container { padding: 1.5rem 2.5rem 3rem 2.5rem; max-width: 1280px; }

/* Texto padrão Streamlit → claro */
p, span, label, div, h1, h2, h3, h4, h5, h6, li, td, th {
    color: #E6EDF3;
}

/* ── Hero ─────────────────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #010409 0%, #0D1117 40%, #161B22 70%, #1C2128 100%);
    border: 1px solid #30363D;
    border-radius: 20px; padding: 3rem 3.5rem; color: white;
    margin-bottom: 2rem;
    box-shadow: 0 0 0 1px #21262D, 0 16px 48px rgba(0,0,0,.6), 0 0 80px rgba(88,166,255,.06);
    position: relative; overflow: hidden;
}
.hero::before {
    content:""; position:absolute; top:-60px; right:-60px;
    width:300px; height:300px; border-radius:50%;
    background: radial-gradient(circle, rgba(88,166,255,.08) 0%, transparent 70%);
}
.hero::after {
    content:""; position:absolute; bottom:-80px; left:25%;
    width:400px; height:400px; border-radius:50%;
    background: radial-gradient(circle, rgba(63,185,80,.04) 0%, transparent 70%);
}
.hero-badge {
    display:inline-block; background:rgba(88,166,255,.15);
    border:1px solid rgba(88,166,255,.3); border-radius:20px;
    padding:4px 14px; font-size:.78rem; font-weight:500;
    letter-spacing:.5px; margin-bottom:1rem; color:#58A6FF;
}
.hero-title { font-size:2.4rem; font-weight:800; letter-spacing:-.5px; margin:0 0 .5rem 0; color:#E6EDF3; }
.hero-sub   { font-size:1.05rem; font-weight:300; opacity:.7; margin:0; color:#8B949E; }

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #161B22 !important; border-radius:14px !important;
    padding:6px !important; gap:4px !important;
    box-shadow: 0 0 0 1px #30363D !important;
    border:none !important; margin-bottom:1.5rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius:10px !important; padding:9px 20px !important;
    font-weight:500 !important; font-size:.88rem !important;
    color:#8B949E !important; border:none !important;
    background:transparent !important; transition:all .2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1F6FEB, #388BFD) !important;
    color: white !important;
    box-shadow: 0 4px 16px rgba(31,111,235,.4) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding:0 !important; }

/* ── Section title ────────────────────────────────────────────────────────── */
.section-title { font-size:1.55rem; font-weight:700; color:#E6EDF3; margin:0 0 .25rem 0; }
.section-title span { color:#58A6FF; }
.section-line { width:48px; height:4px; background:linear-gradient(90deg,#1F6FEB,#58A6FF); border-radius:4px; margin-bottom:1.5rem; }

/* ── Cards ────────────────────────────────────────────────────────────────── */
.card {
    background: #161B22; border-radius:16px; padding:1.6rem 1.8rem;
    box-shadow: 0 0 0 1px #21262D; margin-bottom:1rem;
    border: 1px solid #30363D;
}
.card-accent { border-left:4px solid #58A6FF; }

/* ── KPI Cards ────────────────────────────────────────────────────────────── */
.kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin:1.2rem 0 1.8rem 0; }
.kpi-card {
    background: #161B22; border-radius:14px; padding:1.3rem 1.5rem;
    box-shadow: 0 0 0 1px #21262D; border-top:4px solid #58A6FF; text-align:center;
}
.kpi-card.red   { border-top-color:#FF7B72; }
.kpi-card.amber { border-top-color:#E3B341; }
.kpi-card.green { border-top-color:#3FB950; }
.kpi-value { font-size:1.9rem; font-weight:800; color:#E6EDF3; display:block; line-height:1.1; }
.kpi-label { font-size:.78rem; font-weight:500; color:#8B949E; text-transform:uppercase; letter-spacing:.6px; margin-top:6px; display:block; }

/* ── Stat boxes ───────────────────────────────────────────────────────────── */
.stat-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin:1.2rem 0; }
.stat-box {
    background: #161B22; border:1px solid #30363D;
    border-radius:14px; padding:1.4rem 1.5rem; color:white; text-align:center;
}
.stat-box .stat-val  { font-size:2rem; font-weight:800; color:#58A6FF; display:block; }
.stat-box .stat-desc { font-size:.8rem; color:#8B949E; margin-top:4px; display:block; }

/* ── Insight boxes ────────────────────────────────────────────────────────── */
.insight {
    background: rgba(31,111,235,.08);
    border-left:4px solid #1F6FEB; border-radius:0 12px 12px 0;
    padding:1rem 1.4rem; margin:1rem 0; font-size:.92rem; color:#C9D1D9;
}
.insight strong { color:#58A6FF; }
.insight code.inline { background:rgba(88,166,255,.1); color:#58A6FF; }
.warn-box {
    background: rgba(227,179,65,.08);
    border-left:4px solid #E3B341; border-radius:0 12px 12px 0;
    padding:1rem 1.4rem; margin:1rem 0; font-size:.92rem; color:#C9D1D9;
}
.warn-box strong { color:#E3B341; }
.success-box {
    background: rgba(63,185,80,.08);
    border-left:4px solid #3FB950; border-radius:0 12px 12px 0;
    padding:1rem 1.4rem; margin:1rem 0; font-size:.92rem; color:#C9D1D9;
}
.success-box strong { color:#3FB950; }

/* ── Step list ────────────────────────────────────────────────────────────── */
.step-list { list-style:none; padding:0; margin:0; }
.step-item {
    display:flex; align-items:flex-start; gap:1rem; padding:1rem 1.2rem;
    background: #161B22; border-radius:12px; margin-bottom:.6rem;
    border: 1px solid #21262D;
}
.step-num {
    background: linear-gradient(135deg,#1F6FEB,#388BFD); color:white;
    font-weight:700; font-size:.85rem; width:30px; height:30px;
    border-radius:50%; display:flex; align-items:center; justify-content:center; flex-shrink:0;
}
.step-body { flex:1; }
.step-title { font-weight:600; color:#E6EDF3; font-size:.95rem; }
.step-desc  { color:#8B949E; font-size:.85rem; margin-top:3px; }
code.inline {
    background: rgba(88,166,255,.12); border-radius:4px; padding:1px 6px;
    font-size:.82rem; color:#58A6FF; font-family:'JetBrains Mono','Fira Code',monospace;
}

/* ── Rationale cards ──────────────────────────────────────────────────────── */
.rationale-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin:1rem 0 1.5rem 0; }
.rationale-card {
    background: #161B22; border-radius:14px; padding:1.4rem;
    border:1px solid #30363D; border-top:4px solid #58A6FF;
}
.rationale-card.ridge { border-top-color:#E3B341; }
.rationale-card.lasso { border-top-color:#3FB950; }
.rationale-name  { font-weight:700; color:#E6EDF3; font-size:.95rem; margin-bottom:.4rem; }
.rationale-why   { font-size:.82rem; color:#8B949E; line-height:1.6; }
.rationale-badge {
    display:inline-block; background:rgba(88,166,255,.15); color:#58A6FF;
    border-radius:8px; padding:2px 8px; font-size:.72rem; font-weight:600; margin-bottom:.5rem;
}
.rationale-badge.ridge { background:rgba(227,179,65,.15); color:#E3B341; }
.rationale-badge.lasso { background:rgba(63,185,80,.15);  color:#3FB950; }

/* ── Model table ──────────────────────────────────────────────────────────── */
.model-table { width:100%; border-collapse:separate; border-spacing:0; border-radius:12px; overflow:hidden; border:1px solid #30363D; margin:1rem 0 1.5rem 0; }
.model-table th {
    background: linear-gradient(135deg,#161B22,#1C2128);
    color:#8B949E; font-weight:600; font-size:.78rem; text-transform:uppercase;
    letter-spacing:.5px; padding:12px 16px; text-align:center;
    border-bottom:1px solid #30363D;
}
.model-table td { padding:11px 16px; text-align:center; font-size:.9rem; color:#C9D1D9; background:#161B22; border-bottom:1px solid #21262D; }
.model-table tr:last-child td { border-bottom:none; }
.model-table tr.best-row td { background:rgba(31,111,235,.12); font-weight:700; color:#58A6FF; }
.badge-best { background:linear-gradient(135deg,#1F6FEB,#388BFD); color:white; border-radius:20px; padding:2px 10px; font-size:.72rem; font-weight:600; margin-left:6px; }

/* ── P-value table ────────────────────────────────────────────────────────── */
.pval-table { width:100%; border-collapse:separate; border-spacing:0; border-radius:12px; overflow:hidden; border:1px solid #30363D; margin:1rem 0; }
.pval-table th { background:#1C2128; color:#8B949E; font-weight:600; font-size:.78rem; text-transform:uppercase; letter-spacing:.5px; padding:10px 14px; text-align:left; border-bottom:1px solid #30363D; }
.pval-table td { padding:9px 14px; font-size:.86rem; color:#C9D1D9; background:#161B22; border-bottom:1px solid #21262D; }
.pval-table tr:last-child td { border-bottom:none; }
.pval-table tr.sig td { background:rgba(63,185,80,.06); }
.pval-table tr.not-sig td { background:rgba(255,123,114,.06); }
.sig-badge     { background:rgba(63,185,80,.2);   color:#3FB950; border-radius:6px; padding:2px 8px; font-size:.72rem; font-weight:600; }
.not-sig-badge { background:rgba(255,123,114,.2); color:#FF7B72; border-radius:6px; padding:2px 8px; font-size:.72rem; font-weight:600; }

/* ── Chart card ───────────────────────────────────────────────────────────── */
.chart-card { background:#161B22; border-radius:16px; padding:1.4rem 1.6rem .5rem 1.6rem; border:1px solid #21262D; margin-bottom:1rem; }
.chart-title   { font-size:1rem; font-weight:600; color:#E6EDF3; margin-bottom:.2rem; }
.chart-caption { font-size:.78rem; color:#8B949E; margin-top:-.3rem; margin-bottom:.6rem; }

/* ── Streamlit overrides ──────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg,#1F6FEB,#388BFD) !important; color:white !important;
    border:none !important; border-radius:10px !important; padding:.7rem 2rem !important;
    font-weight:600 !important; font-size:.95rem !important;
    box-shadow:0 4px 15px rgba(31,111,235,.4) !important; transition:all .25s ease !important; width:100%;
}
.stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 6px 22px rgba(31,111,235,.55) !important; }

div[data-testid="stMetricValue"] { font-size:1.8rem !important; font-weight:800 !important; color:#E6EDF3 !important; }
div[data-testid="stMetricLabel"] { font-size:.78rem !important; font-weight:500 !important; color:#8B949E !important; text-transform:uppercase; letter-spacing:.5px; }

/* Selectbox — borda customizada (cores base vêm do config.toml dark theme) */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    border-color:#30363D !important; border-radius:10px !important;
}

/* Slider */
.stSlider > div > div > div > div { background:#1F6FEB !important; }
[data-testid="stSlider"] > div > div { background:#21262D !important; }

/* Expander */
div[data-testid="stExpander"] {
    background:#161B22 !important; border-radius:12px !important;
    border:1px solid #30363D !important; margin-bottom:.5rem;
}
div[data-testid="stExpander"] summary { color:#E6EDF3 !important; }

/* DataFrames */
[data-testid="stDataFrame"] { background:#161B22 !important; }

/* Inputs dark */
input, textarea, select {
    background:#161B22 !important; color:#E6EDF3 !important; border-color:#30363D !important;
}

/* ── Legenda de Cores (Gestalt: Similaridade) ─────────────────────────────── */
.color-legend {
    background:#161B22; border:1px solid #30363D; border-radius:12px;
    padding:.8rem 1.4rem; margin-bottom:1.5rem;
    display:flex; align-items:center; gap:2rem; flex-wrap:wrap;
}
.legend-title { font-size:.72rem; font-weight:600; color:#8B949E; text-transform:uppercase; letter-spacing:.6px; }
.legend-items { display:flex; gap:1.5rem; flex-wrap:wrap; }
.legend-item  { display:flex; align-items:center; gap:6px; font-size:.82rem; color:#C9D1D9; }
.legend-dot   { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
.legend-dot.blue  { background:#58A6FF; }
.legend-dot.red   { background:#FF7B72; }
.legend-dot.amber { background:#E3B341; }
.legend-dot.green { background:#3FB950; }

/* ── Footer ───────────────────────────────────────────────────────────────── */
.footer { text-align:center; padding:2rem 0 1rem 0; color:#8B949E; font-size:.78rem; border-top:1px solid #21262D; margin-top:2rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODELS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Carregando dados e treinando modelos…")
def get_all():
    raw   = load_data()
    clean = preprocess(raw)
    X, y  = build_features(clean)
    results, trained, X_cols = train_models(X, y)
    return raw, clean, X, y, results, trained, X_cols

@st.cache_data(show_spinner="Calculando p-values (statsmodels OLS)…")
def get_pvals(_X, _y):
    return get_pvalues(_X, _y)

raw_df, clean_df, X, y, model_results, trained_models, X_cols = get_all()
pval_coefs, pval_pvals = get_pvals(X, y)

best_name  = max(model_results, key=lambda k: model_results[k]["R²"])
best_pipe  = trained_models[best_name][0]
best_coefs = pd.Series(best_pipe.named_steps["model"].coef_, index=X_cols)

lasso_pipe  = trained_models["Lasso (L1)"][0]
lasso_coefs = pd.Series(lasso_pipe.named_steps["model"].coef_, index=X_cols)

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge">TEMA 1 — REGRESSÃO · AVD + MACHINE LEARNING</div>
    <p class="hero-title">🏠 Precificação Inteligente de Imóveis<br>em Bengaluru</p>
    <p class="hero-sub">
        Transformamos 13.000 registros históricos em uma régua inteligente de precificação.<br>
        Regressão Linear · Ridge · Lasso · Scikit-learn · Plotly
    </p>
</div>
""", unsafe_allow_html=True)

# Gestalt: Similaridade — mesma cor = mesmo significado em TODOS os gráficos
st.markdown("""
<div class="color-legend">
    <span class="legend-title">Gestalt · Similaridade — Legenda global de cores</span>
    <div class="legend-items">
        <div class="legend-item"><div class="legend-dot blue"></div>Dados reais / Features</div>
        <div class="legend-item"><div class="legend-dot red"></div>Previsão / Tendência / Referência</div>
        <div class="legend-item"><div class="legend-dot amber"></div>Erros / Resíduos / Alertas</div>
        <div class="legend-item"><div class="legend-dot green"></div>Selecionado / Positivo / Boa oportunidade</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖  A Dor do Negócio",
    "🔍  EDA",
    "⚙️  Pré-processamento",
    "📊  Modelos & Avaliação",
    "🔮  Simulador de Preço",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-title">O <span>Problema</span></p><div class="section-line"></div>', unsafe_allow_html=True)

    col_txt, col_sol = st.columns([3, 2], gap="large")

    with col_txt:
        st.markdown("""
        <div class="card card-accent">
            <p style="color:#C9D1D9;font-size:1rem;line-height:1.75;margin:0;">
                Bengaluru — a <em>Silicon Valley</em> da Índia — é uma das cidades que mais cresceu
                nas últimas décadas. Com isso, o mercado imobiliário tornou-se <strong>caótico e opaco</strong>:
                compradores pagam a mais, vendedores precificam sem referência e corretoras perdem
                negócios por falta de dados confiáveis.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### O que aconteceu em 2017?")
        st.markdown("""
        <div class="stat-grid">
            <div class="stat-box"><span class="stat-val">−7%</span><span class="stat-desc">Queda nas vendas de imóveis em toda a Índia</span></div>
            <div class="stat-box"><span class="stat-val">−5%</span><span class="stat-desc">Queda de preços em Bengaluru no 2º semestre</span></div>
            <div class="stat-box"><span class="stat-val">9k+</span><span class="stat-desc">Apartamentos disponíveis só na faixa ₹42–52L</span></div>
        </div>
        <div class="insight">
            <strong>Causas:</strong> desmonetização, RERA e desconfiança nos incorporadores.<br><br>
            <em>"Com mais de 9.000 apartamentos disponíveis em uma mesma faixa de preço,
            como saber se o valor pedido é justo?"</em>
        </div>
        """, unsafe_allow_html=True)

    with col_sol:
        st.markdown("""
        <div class="card" style="height:100%;">
            <p style="font-weight:700;color:#E6EDF3;font-size:1rem;margin:0 0 .8rem 0;">💡 Nossa Solução</p>
            <p style="color:#8B949E;font-size:.9rem;margin:0 0 1rem 0;">
                Um modelo de <strong style="color:#C9D1D9;">Regressão supervisionada</strong> treinado com dados históricos
                reais de Bengaluru.
            </p>
            <hr style="border:none;border-top:1px solid #30363D;margin:.8rem 0;">
            <p style="font-weight:600;color:#8B949E;font-size:.85rem;margin:0 0 .4rem 0;text-transform:uppercase;letter-spacing:.5px;">PERGUNTA DO MODELO</p>
            <p style="color:#58A6FF;font-size:.9rem;font-style:italic;margin:0 0 1rem 0;">
                "Dado o tamanho, localização, quartos e banheiros — qual o preço justo?"
            </p>
            <hr style="border:none;border-top:1px solid #30363D;margin:.8rem 0;">
            <p style="font-weight:600;color:#8B949E;font-size:.85rem;margin:0 0 .6rem 0;text-transform:uppercase;letter-spacing:.5px;">FLUXO DO PROJETO</p>
        """ + "".join([
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
            f'<div style="width:24px;height:24px;border-radius:50%;background:linear-gradient(135deg,#2E86AB,#1A5276);'
            f'color:white;font-size:.75rem;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0;">{i}</div>'
            f'<span style="font-size:.85rem;color:#C9D1D9;">{s}</span></div>'
            for i, s in enumerate(["Coleta & Limpeza","Análise Exploratória (EDA)","Pré-processamento","Treinamento & Avaliação","Simulador Interativo"], 1)
        ]) + "</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — EDA
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-title">Análise <span>Exploratória</span> de Dados</p><div class="section-line"></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card"><span class="kpi-value">{len(raw_df):,}</span><span class="kpi-label">Registros Brutos</span></div>
        <div class="kpi-card green"><span class="kpi-value">{len(clean_df):,}</span><span class="kpi-label">Após Limpeza</span></div>
        <div class="kpi-card amber"><span class="kpi-value">{clean_df['location'].nunique()}</span><span class="kpi-label">Localizações</span></div>
        <div class="kpi-card"><span class="kpi-value">₹ {clean_df['price'].mean():.0f}L</span><span class="kpi-label">Preço Médio</span></div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="medium")

    with col_l:
        # Gestalt: Ponto Focal — linha da mediana em AMBER destaca o valor central
        st.markdown('<div class="chart-card"><div class="chart-title">Distribuição de Preços</div><div class="chart-caption">Maioria dos imóveis entre ₹40L–₹150L; cauda longa à direita</div>', unsafe_allow_html=True)
        median_price = clean_df["price"].median()
        fig = px.histogram(clean_df, x="price", nbins=60, color_discrete_sequence=[BLUE],
                           labels={"price": "Preço (Lakhs ₹)", "count": "Nº de Imóveis"})
        fig = plotly_defaults(fig)
        fig.update_layout(xaxis_title="Preço (Lakhs ₹)", yaxis_title="Nº de Imóveis", bargap=0.04)
        fig.update_traces(marker_line_width=0, opacity=0.85)
        # Ponto Focal: mediana em AMBER (cor de alerta = valor de referência)
        fig.add_vline(x=median_price, line_color=AMBER, line_width=2, line_dash="dash")
        fig.add_annotation(x=median_price, y=1, yref="paper", yanchor="top",
                           text=f" Mediana<br> ₹{median_price:.0f}L",
                           showarrow=False, font=dict(color=AMBER, size=11),
                           xanchor="left", bgcolor="rgba(22,27,34,.8)",
                           bordercolor=AMBER, borderwidth=1)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        # Gestalt: Continuidade — trendline em RED guia o olho pela relação linear
        st.markdown('<div class="chart-card"><div class="chart-title">Área Total vs. Preço</div><div class="chart-caption">Relação linear positiva confirmada pela regressão OLS</div>', unsafe_allow_html=True)
        sample = clean_df.sample(min(2000, len(clean_df)), random_state=42)
        pearson_r = clean_df[["total_sqft", "price"]].corr().iloc[0, 1]
        fig2 = px.scatter(
            sample, x="total_sqft", y="price",
            trendline="ols", trendline_color_override=RED,
            color_discrete_sequence=[BLUE], opacity=0.35,
            labels={"total_sqft": "Área Total (sqft)", "price": "Preço (Lakhs ₹)"},
        )
        fig2 = plotly_defaults(fig2, show_legend=False)
        # Proximidade: anotação junto à linha de tendência
        fig2.add_annotation(
            x=0.04, y=0.94, xref="paper", yref="paper",
            text=f"r = {pearson_r:.2f}", showarrow=False, xanchor="left",
            bgcolor="rgba(22,27,34,.85)", bordercolor=RED, borderwidth=1,
            font=dict(color=RED, size=12))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Gestalt: Proximidade — rótulos diretamente nas barras (sem legenda de eixo X)
    st.markdown('<div class="chart-card"><div class="chart-title">Top 15 Localizações — Preço Médio por sqft</div><div class="chart-caption">Bairros premium chegam a 5× a média da cidade — localização é o driver principal</div>', unsafe_allow_html=True)
    top_loc = (clean_df.groupby("location")["price_per_sqft"].mean()
               .sort_values(ascending=False).head(15).reset_index())
    city_avg = clean_df["price_per_sqft"].mean()
    fig3 = go.Figure(go.Bar(
        x=top_loc["price_per_sqft"], y=top_loc["location"], orientation="h",
        marker_color=[BLUE if v >= city_avg * 2 else "#2D5A8E" for v in top_loc["price_per_sqft"]],
        marker_line_width=0,
        text=[f"₹{v:,.0f}" for v in top_loc["price_per_sqft"]],
        textposition="outside", textfont=dict(color="#8B949E", size=10),
    ))
    fig3 = plotly_defaults(fig3, ygrid=False)
    fig3.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="",
                       xaxis_showticklabels=False)
    # Ponto Focal: linha da média da cidade
    fig3.add_vline(x=city_avg, line_color=AMBER, line_width=1, line_dash="dot",
                   annotation_text=f" Média cidade ₹{city_avg:,.0f}",
                   annotation_font=dict(color=AMBER, size=10))
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Gestalt: Similaridade — escala de cor única (Blues) = intensidade de correlação
    st.markdown('<div class="chart-card"><div class="chart-title">Correlação entre Variáveis Numéricas</div><div class="chart-caption">total_sqft (0.54) e bath (0.52) são as features mais correlacionadas com price</div>', unsafe_allow_html=True)
    corr = clean_df[["total_sqft", "bath", "bhk", "balcony", "price"]].corr()
    # Fix 1 — Similaridade: escala divergente RED→neutro→BLUE
    # correlação negativa = RED, zero = neutro escuro, positiva = BLUE
    fig4 = px.imshow(corr, text_auto=".2f",
                     color_continuous_scale=[[0, "#FF7B72"], [0.5, "#1C2128"], [1, "#58A6FF"]],
                     labels={"color": "r de Pearson"}, zmin=-1, zmax=1)
    fig4 = plotly_defaults(fig4)
    fig4.update_layout(coloraxis_colorbar=dict(
        tickfont=dict(color="#8B949E"), title_font=dict(color="#8B949E")))
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Pré-processamento
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-title">Pipeline de <span>Pré-processamento</span></p><div class="section-line"></div>', unsafe_allow_html=True)

    steps = [
        ("Remoção de coluna irrelevante",  "<code class='inline'>society</code> — 41% de valores nulos, sem poder preditivo."),
        ("Extração de BHK",               "<code class='inline'>'3 BHK'</code> → <code class='inline'>3</code> (inteiro extraído do texto)."),
        ("Conversão de total_sqft",        "<code class='inline'>'1195–1440'</code> → <code class='inline'>1317.5</code> (média do intervalo)."),
        ("Tratamento de nulos",            "Linhas sem <code class='inline'>bath</code>, <code class='inline'>bhk</code>, <code class='inline'>price</code> ou <code class='inline'>location</code> são removidas."),
        ("Agrupamento de localização",     "1.287 bairros → ~241 (bairros com &lt;10 registros viram <code class='inline'>'other'</code>)."),
        ("Remoção de outliers",            "sqft/BHK &lt; 300 | price_per_sqft fora de ±1σ por bairro | bath &gt; BHK + 2."),
        ("One-Hot Encoding",               "Coluna <code class='inline'>location</code> transformada em variáveis binárias (dummies)."),
        # CORREÇÃO 2: normalização explícita
        ("Normalização — StandardScaler",  "Todas as features são padronizadas (μ=0, σ=1) dentro do sklearn <code class='inline'>Pipeline</code>, garantindo que Ridge e Lasso penalizem coeficientes na mesma escala."),
    ]

    html_steps = '<ul class="step-list">'
    for i, (title, desc) in enumerate(steps, 1):
        html_steps += f"""
        <li class="step-item">
            <div class="step-num">{i}</div>
            <div class="step-body"><div class="step-title">{title}</div><div class="step-desc">{desc}</div></div>
        </li>"""
    html_steps += "</ul>"
    st.markdown(html_steps, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l2, col_r2 = st.columns(2, gap="medium")

    with col_l2:
        st.markdown('<div class="chart-card"><div class="chart-title">Antes vs. Depois da Limpeza</div>', unsafe_allow_html=True)
        comparison = pd.DataFrame({
            "Etapa":  ["Registros totais", "Localizações únicas", "Outliers de preço"],
            "Antes":  [f"{len(raw_df):,}", "1.287", "Presentes"],
            "Depois": [f"{len(clean_df):,}", str(clean_df['location'].nunique()), "Removidos"],
        })
        st.dataframe(comparison, hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r2:
        # Gestalt: Similaridade — 2 e 3 BHK em BLUE cheio, demais em tom recuado
        st.markdown('<div class="chart-card"><div class="chart-title">Distribuição de BHK (após limpeza)</div><div class="chart-caption">2 e 3 BHK dominam o mercado — destaque em azul</div>', unsafe_allow_html=True)
        bhk_counts = clean_df["bhk"].value_counts().sort_index().reset_index()
        bhk_counts.columns = ["BHK", "Imóveis"]
        # Fix 5 — Figura-Fundo: cor secundária mais visível sobre fundo escuro
        bar_colors = [BLUE if b in [2, 3] else "#2D5F8A" for b in bhk_counts["BHK"]]
        fig_bhk = go.Figure(go.Bar(
            x=bhk_counts["BHK"], y=bhk_counts["Imóveis"],
            marker_color=bar_colors, marker_line_width=0,
            text=bhk_counts["Imóveis"], textposition="outside",
            textfont=dict(color="#8B949E", size=11),
        ))
        fig_bhk = plotly_defaults(fig_bhk)
        fig_bhk.update_layout(xaxis=dict(tickmode="linear", tickfont=dict(color="#8B949E")),
                              showlegend=False, yaxis_title="Quantidade", xaxis_title="Quartos (BHK)")
        st.plotly_chart(fig_bhk, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # CORREÇÃO 2: Lasso feature selection explícita
    st.markdown("---")
    st.markdown('<p class="section-title">Feature <span>Selection</span> pelo Lasso</p><div class="section-line"></div>', unsafe_allow_html=True)

    n_total  = len(lasso_coefs)
    n_zeroed = int((lasso_coefs == 0).sum())
    n_kept   = n_total - n_zeroed

    st.markdown(f"""
    <div class="insight">
        O Lasso (L1) penaliza coeficientes empurrando-os a zero — eliminando automaticamente variáveis
        irrelevantes. Das <strong>{n_total} features</strong> após o encoding, o Lasso zerou
        <strong>{n_zeroed} ({n_zeroed/n_total*100:.0f}%)</strong> e manteve
        <strong>{n_kept}</strong> como relevantes para o modelo.
    </div>
    """, unsafe_allow_html=True)

    col_fs1, col_fs2 = st.columns([1, 2], gap="medium")

    with col_fs1:
        st.markdown(f"""
        <div class="kpi-grid" style="grid-template-columns:repeat(1,1fr);">
            <div class="kpi-card"><span class="kpi-value">{n_total}</span><span class="kpi-label">Features totais</span></div>
            <div class="kpi-card green"><span class="kpi-value">{n_kept}</span><span class="kpi-label">Mantidas pelo Lasso</span></div>
            <div class="kpi-card red"><span class="kpi-value">{n_zeroed}</span><span class="kpi-label">Zeradas (descartadas)</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_fs2:
        # Gestalt: Similaridade — GREEN = selecionado/mantido (consistente com resto do app)
        st.markdown('<div class="chart-card"><div class="chart-title">Top 15 Features Mantidas pelo Lasso</div><div class="chart-caption">Apenas estas variáveis tiveram coeficiente ≠ 0 após regularização L1</div>', unsafe_allow_html=True)
        kept_coefs = lasso_coefs[lasso_coefs != 0].abs().sort_values(ascending=False).head(15)
        fig_lasso = go.Figure(go.Bar(
            x=kept_coefs.values, y=kept_coefs.index, orientation="h",
            marker_color=GREEN, marker_line_width=0,
            text=[f"{v:.2f}" for v in kept_coefs.values],
            textposition="outside", textfont=dict(color="#8B949E", size=10),
        ))
        fig_lasso = plotly_defaults(fig_lasso, ygrid=False)
        fig_lasso.update_layout(yaxis={"categoryorder": "total ascending"},
                                xaxis_title="|Coeficiente Lasso|", xaxis_showticklabels=False)
        st.plotly_chart(fig_lasso, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — Modelos & Avaliação
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-title">Treinamento e <span>Avaliação</span> dos Modelos</p><div class="section-line"></div>', unsafe_allow_html=True)

    # CORREÇÃO 3: Justificativa técnica dos modelos
    st.markdown("### Por que esses algoritmos?")
    st.markdown("""
    <div class="rationale-grid">
        <div class="rationale-card">
            <div class="rationale-badge">BASELINE</div>
            <div class="rationale-name">📈 Regressão Linear</div>
            <div class="rationale-why">
                Ponto de partida interpretável: os coeficientes mostram diretamente
                quanto cada variável move o preço em Lakhs. Serve como referência
                para avaliar se modelos mais complexos realmente melhoram.
            </div>
        </div>
        <div class="rationale-card ridge">
            <div class="rationale-badge ridge">REGULARIZAÇÃO L2</div>
            <div class="rationale-name">🛡️ Ridge</div>
            <div class="rationale-why">
                Com 241 features (dummies de localização), a Regressão Linear pode
                overfitar. O Ridge penaliza coeficientes grandes, distribuindo o
                peso entre variáveis correlacionadas — ideal para alta dimensionalidade.
            </div>
        </div>
        <div class="rationale-card lasso">
            <div class="rationale-badge lasso">REGULARIZAÇÃO L1</div>
            <div class="rationale-name">✂️ Lasso</div>
            <div class="rationale-why">
                Vai além do Ridge: zera coeficientes de variáveis irrelevantes,
                fazendo <strong>feature selection automática</strong>. Útil para
                identificar quais bairros realmente importam para o preço.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight">
        Divisão: <strong>80% treino / 20% teste</strong> com validação cruzada 5-fold dentro de um
        sklearn <code class="inline">Pipeline</code> (StandardScaler → Model), evitando data leakage.
    </div>
    """, unsafe_allow_html=True)

    # Tabela de resultados
    st.markdown("### Comparação de Resultados")
    results_df = pd.DataFrame(model_results).T.reset_index().rename(columns={"index": "Modelo"})
    mae_best   = model_results[best_name]["MAE (Lakhs)"]

    rows_html = ""
    for _, row in results_df.iterrows():
        is_best = row["Modelo"] == best_name
        cls   = 'class="best-row"' if is_best else ""
        badge = '<span class="badge-best">★ Melhor</span>' if is_best else ""
        rows_html += f"""
        <tr {cls}>
            <td style="text-align:left;font-weight:{'700' if is_best else '400'}">{row['Modelo']}{badge}</td>
            <td>{row['R²']}</td><td>{row['MAE (Lakhs)']}</td>
            <td>{row['RMSE (Lakhs)']}</td><td>{row['CV R² (5-fold)']}</td>
        </tr>"""

    st.markdown(f"""
    <table class="model-table">
        <thead><tr>
            <th style="text-align:left">Modelo</th><th>R²</th>
            <th>MAE (Lakhs)</th><th>RMSE (Lakhs)</th><th>CV R² (5-fold)</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    <div class="insight">
        <strong>MAE — {best_name}:</strong> em média, o modelo erra <strong>₹ {mae_best} Lakhs</strong>
        por previsão. Para um imóvel de ₹ 50L → margem de <strong>{mae_best/50*100:.1f}%</strong>.
        Para um corretor, isso é aceitável como referência de mercado.
    </div>
    """, unsafe_allow_html=True)

    # CORREÇÃO 1: P-values
    st.markdown("### Significância Estatística — P-values (OLS)")
    st.markdown("""
    <div class="insight">
        P-value mede a probabilidade de um coeficiente ser zero por acaso.
        <strong>p &lt; 0.05</strong> → variável estatisticamente significativa para o preço.
        Os valores abaixo usam OLS com features normalizadas (statsmodels).
    </div>
    """, unsafe_allow_html=True)

    numeric_feats = ["total_sqft", "bath", "bhk"]
    top_loc_feats = pval_coefs.drop(labels=numeric_feats, errors="ignore").abs().sort_values(ascending=False).head(12).index.tolist()
    show_feats    = numeric_feats + top_loc_feats

    pval_rows = ""
    for feat in show_feats:
        coef = pval_coefs.get(feat, 0)
        pval = pval_pvals.get(feat, 1)
        sig  = pval < 0.05
        cls  = "sig" if sig else "not-sig"
        badge = f'<span class="sig-badge">✓ Significativo</span>' if sig else f'<span class="not-sig-badge">✗ Não significativo</span>'
        pval_rows += f"""
        <tr class="{cls}">
            <td><code class="inline">{feat}</code></td>
            <td>{coef:+.4f}</td>
            <td>{pval:.4f}</td>
            <td>{badge}</td>
        </tr>"""

    st.markdown(f"""
    <table class="pval-table">
        <thead><tr>
            <th>Variável</th><th>Coeficiente (std)</th><th>P-value</th><th>Significância</th>
        </tr></thead>
        <tbody>{pval_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    # Scatter Reais vs Previstos + Resíduos
    st.markdown("### Diagnóstico do Modelo")
    _, y_test_arr, y_pred_arr = trained_models[best_name]
    residuals = y_pred_arr - y_test_arr

    col_a, col_b = st.columns(2, gap="medium")

    with col_a:
        # Gestalt: Figura-Fundo — pontos BLUE (dados) sobre diagonal RED (referência)
        # Gestalt: Ponto Focal — R² anotado junto ao gráfico (Proximidade)
        st.markdown(f'<div class="chart-card"><div class="chart-title">Reais vs. Previstos — {best_name}</div><div class="chart-caption">Pontos sobre a diagonal = previsão perfeita; faixa AMBER = margem do MAE</div>', unsafe_allow_html=True)
        r2_val  = model_results[best_name]["R²"]
        mae_val = model_results[best_name]["MAE (Lakhs)"]
        max_val = float(max(y_test_arr.max(), y_pred_arr.max()))

        fig_rv = go.Figure()
        # Faixa MAE (Figura-Fundo: zona aceitável em plano de fundo)
        fig_rv.add_trace(go.Scatter(
            x=[0, max_val], y=[mae_val, max_val + mae_val], mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_rv.add_trace(go.Scatter(
            x=[0, max_val], y=[-mae_val, max_val - mae_val], mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(227,179,65,.16)", name=f"±MAE ({mae_val}L)",
            hoverinfo="skip"))
        # Diagonal de referência (Continuidade)
        fig_rv.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val], mode="lines",
            line=dict(color=RED, dash="dash", width=1.5), name="Previsão perfeita"))
        # Dados reais (Figura)
        fig_rv.add_trace(go.Scatter(
            x=y_test_arr, y=y_pred_arr, mode="markers",
            marker=dict(color=BLUE, opacity=0.35, size=4), name="Imóveis", showlegend=False))
        fig_rv = plotly_defaults(fig_rv, show_legend=True)
        fig_rv.update_layout(xaxis_title="Preço Real (Lakhs ₹)", yaxis_title="Preço Previsto (Lakhs ₹)")
        # Proximidade: R² anotado no canto superior esquerdo
        fig_rv.add_annotation(
            x=0.04, y=0.95, xref="paper", yref="paper",
            text=f"R² = {r2_val:.4f}", showarrow=False, xanchor="left",
            bgcolor="rgba(22,27,34,.9)", bordercolor=BLUE, borderwidth=1,
            font=dict(color=BLUE, size=13))
        st.plotly_chart(fig_rv, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        # Gestalt: Ponto Focal — zero (μ=0 ideal) em GREEN; ±1σ em cinza como referência
        st.markdown('<div class="chart-card"><div class="chart-title">Distribuição dos Resíduos</div><div class="chart-caption">Distribuição centrada em zero = modelo sem viés; ±1σ delimitam 68% dos erros</div>', unsafe_allow_html=True)
        std_r  = float(np.std(residuals))
        mean_r = float(np.mean(residuals))
        fig_res = px.histogram(x=residuals, nbins=60, color_discrete_sequence=[AMBER],
                               labels={"x": "Resíduo (Lakhs ₹)", "count": "Frequência"})
        fig_res = plotly_defaults(fig_res)
        fig_res.update_layout(xaxis_title="Resíduo (Previsto − Real, Lakhs ₹)", yaxis_title="Frequência")
        fig_res.update_traces(marker_line_width=0, opacity=0.8)
        # Focal Point: zero em GREEN (ideal)
        fig_res.add_vline(x=0, line_color=GREEN, line_width=2)
        fig_res.add_annotation(x=0, y=1, yref="paper", yanchor="top",
                               text=" μ=0<br> ideal", showarrow=False,
                               font=dict(color=GREEN, size=11), xanchor="left",
                               bgcolor="rgba(22,27,34,.8)", bordercolor=GREEN, borderwidth=1)
        # ±1σ em cinza (referência de dispersão)
        fig_res.add_vline(x=std_r,  line_color="#8B949E", line_width=1, line_dash="dot",
                          annotation_text=f" +1σ ({std_r:.1f}L)",
                          annotation_font=dict(color="#8B949E", size=10),
                          annotation_position="top right")
        fig_res.add_vline(x=-std_r, line_color="#8B949E", line_width=1, line_dash="dot",
                          annotation_text=f"-1σ ",
                          annotation_font=dict(color="#8B949E", size=10),
                          annotation_position="top left")
        st.plotly_chart(fig_res, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Gestalt: Similaridade — BLUE = impacto positivo, RED = impacto negativo
    # Gestalt: Proximidade — rótulos diretamente nas barras
    st.markdown(f'<div class="chart-card"><div class="chart-title">Variáveis Mais Influentes — {best_name}</div><div class="chart-caption">Azul = eleva o preço · Vermelho = reduz o preço · Ordenado por impacto absoluto</div>', unsafe_allow_html=True)
    top15  = best_coefs.abs().sort_values(ascending=False).head(15)
    top_df = pd.DataFrame({"Variável": top15.index, "Coeficiente": best_coefs[top15.index].values})
    bar_colors_coef = [BLUE if c > 0 else RED for c in top_df["Coeficiente"]]
    fig_coef = go.Figure(go.Bar(
        x=top_df["Coeficiente"], y=top_df["Variável"], orientation="h",
        marker_color=bar_colors_coef, marker_line_width=0,
        text=[f"{c:+.1f}L" for c in top_df["Coeficiente"]],
        textposition="outside", textfont=dict(color="#8B949E", size=10),
    ))
    fig_coef = plotly_defaults(fig_coef, ygrid=False)
    # Fix 4 — Redução de Ruído: rótulos nas barras já mostram os valores, eixo X é redundante
    fig_coef.update_layout(yaxis={"categoryorder": "total ascending"},
                           xaxis_title="", xaxis_showticklabels=False, showlegend=False)
    # Linha de zero como referência (Fechamento visual)
    fig_coef.add_vline(x=0, line_color="#30363D", line_width=1.5)
    st.plotly_chart(fig_coef, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — Simulador
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<p class="section-title">🔮 Simulador de <span>Preço</span></p><div class="section-line"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight">
        Configure as características do imóvel e clique em <strong>Calcular</strong>
        para obter a estimativa. O resultado inclui comparação com a mediana do bairro.
    </div>
    """, unsafe_allow_html=True)

    locations_available = sorted(loc for loc in X_cols if loc not in ("total_sqft", "bath", "bhk", "other"))

    col_form, col_chart = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        location = st.selectbox("📍 Localização", locations_available)
        sqft     = st.slider("📐 Área Total (sqft)", 300, 10_000, 1_200, step=50)
        bhk      = st.slider("🛏️ Quartos (BHK)", 1, 10, 2)
        bath     = st.slider("🚿 Banheiros", 1, 10, 2)
        st.markdown("<br>", unsafe_allow_html=True)
        calc = st.button("Calcular Preço Estimado", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if calc:
            price = predict_price(best_pipe, X_cols, location, sqft, bath, bhk)
            st.markdown(f"""
            <div class="card" style="border-top:4px solid #3FB950;margin-top:1rem;">
                <p style="font-size:.78rem;font-weight:600;color:#8A9BB0;text-transform:uppercase;letter-spacing:.6px;margin:0 0 .3rem 0;">PREÇO ESTIMADO</p>
                <p style="font-size:2.2rem;font-weight:800;color:#E6EDF3;margin:0;">₹ {price:.2f} <span style="font-size:1rem;font-weight:400;color:#8B949E;">Lakhs</span></p>
                <p style="font-size:.8rem;color:#8A9BB0;margin:.4rem 0 0 0;">Modelo: {best_name}</p>
            </div>
            """, unsafe_allow_html=True)

            if location in clean_df["location"].values:
                median_p = clean_df[clean_df["location"] == location]["price"].median()
                diff = price - median_p
                if diff > 0:
                    st.markdown(f'<div class="warn-box">⚠️ Este imóvel está <strong>₹ {abs(diff):.1f}L acima</strong> da mediana do bairro (₹ {median_p:.1f}L). Negocie o preço.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="success-box">✅ Este imóvel está <strong>₹ {abs(diff):.1f}L abaixo</strong> da mediana do bairro (₹ {median_p:.1f}L). Boa oportunidade!</div>', unsafe_allow_html=True)

    with col_chart:
        st.markdown('<div class="chart-card" style="height:100%;"><div class="chart-title">Imóveis Similares na Região</div><div class="chart-caption">Comparação com imóveis do mesmo BHK e bairro</div>', unsafe_allow_html=True)
        if location in clean_df["location"].values:
            similar = clean_df[(clean_df["location"] == location) & (clean_df["bhk"] == bhk)]
            if len(similar) >= 5:
                fig_sim = px.scatter(similar, x="total_sqft", y="price",
                                     color_discrete_sequence=[BLUE], opacity=0.7,
                                     labels={"total_sqft": "Área (sqft)", "price": "Preço (Lakhs ₹)"})
                fig_sim = plotly_defaults(fig_sim)
                fig_sim.add_vline(x=sqft, line_dash="dash", line_color=RED,
                                  annotation_text="Seu imóvel", annotation_position="top right")
                fig_sim.update_layout(showlegend=False)
                st.plotly_chart(fig_sim, use_container_width=True)
                st.markdown(f'<p style="font-size:.78rem;color:#8A9BB0;">{len(similar)} imóveis com {bhk} BHK em {location}.</p>', unsafe_allow_html=True)
            else:
                all_loc = clean_df[clean_df["location"] == location]
                fig_box = px.box(all_loc, y="price", x="bhk", color_discrete_sequence=[BLUE],
                                 labels={"price": "Preço (Lakhs ₹)", "bhk": "BHK"})
                fig_box = plotly_defaults(fig_box)
                st.plotly_chart(fig_box, use_container_width=True)
                st.markdown(f'<p style="font-size:.78rem;color:#8A9BB0;">Poucos dados para {bhk} BHK em {location} — exibindo distribuição geral.</p>', unsafe_allow_html=True)
        else:
            st.info("Localização sem dados suficientes para comparação visual.")
        st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Bengaluru House Price Predictor &nbsp;·&nbsp; Tema 1: Regressão &nbsp;·&nbsp;
    AVD + Machine Learning &nbsp;·&nbsp; CESAR School<br>
    Scikit-learn · Statsmodels · Plotly · Streamlit
</div>
""", unsafe_allow_html=True)
