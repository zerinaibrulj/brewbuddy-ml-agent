"""
BrewBuddy - AI + ML hybrid coffee recommender. Modern Streamlit UI.
"""

from __future__ import annotations

import os
import random
from ast import literal_eval
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.linear_model import Ridge

from background_worker import BackgroundWorker
from brewbuddy_agent import BrewBuddyAgent, NoWork
from brewbuddy_data.database import (
    get_user_profile,
    import_dataset_rows,
    init_db,
    save_user_profile,
)

# —— Theme: premium dark, warm gold / espresso accents ——
st.set_page_config(
    page_title="BrewBuddy",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#d4a674"
ACCENT2 = "#b86b3a"
PLOT_FONT = "#e2ddd6"
PLOT_MUTED = "#5c5a55"


def _apply_plot_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14, 14, 20, 0.85)",
        font=dict(color=PLOT_FONT, family="ui-sans-serif, system-ui, sans-serif", size=12),
        title_font=dict(size=15, color=ACCENT),
        xaxis=dict(
            gridcolor="rgba(212, 166, 116, 0.12)", zerolinecolor="rgba(255,255,255,0.05)", showgrid=True
        ),
        yaxis=dict(
            gridcolor="rgba(212, 166, 116, 0.12)", zerolinecolor="rgba(255,255,255,0.05)", showgrid=True
        ),
    )
    return fig


def _to_dict_safe(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            out = literal_eval(value)
            if isinstance(out, dict):
                return out
        except (ValueError, SyntaxError):
            return {}
    return {}


def _to_list_safe(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            out = literal_eval(value)
            if isinstance(out, list):
                return out
        except (ValueError, SyntaxError):
            return []
    return []


def _history_eval_frame(agent: BrewBuddyAgent) -> pd.DataFrame:
    rows = []
    for h in agent.interaction_history:
        subj = h.get("subjective") or {}
        if isinstance(subj, str):
            subj = _to_dict_safe(subj)
        cos = h.get("cosine_scores") or {}
        if isinstance(cos, str):
            cos = _to_dict_safe(cos)
        rows.append(
            {
                "coffee": h.get("coffee"),
                "rating": float(h.get("rating", 0)),
                "ml_state": h.get("ml_state", "unknown"),
                "sleep": float(subj.get("sleep_hours", 7.0)),
                "fatigue": float(subj.get("fatigue", 5.0)),
                "lactose": 1.0 if subj.get("lactose_intolerance", False) else 0.0,
                "social_empty": 1.0 if str(subj.get("social_battery", "Full")) == "Empty" else 0.0,
                "candidates": h.get("candidates", []),
                "cosine_scores": cos,
            }
        )
    return pd.DataFrame(rows)


def _feature_vector(
    coffee_name: str,
    ctx: pd.Series,
    coffee_items: dict,
    cosine_scores: dict,
) -> list:
    meta = coffee_items.get(coffee_name, {})
    return [
        float(ctx.get("sleep", 7.0)) / 12.0,
        float(ctx.get("fatigue", 5.0)) / 10.0,
        float(ctx.get("lactose", 0.0)),
        float(ctx.get("social_empty", 0.0)),
        float(meta.get("caffeine_level", 0.5)),
        float(meta.get("dairy_load", 0.0)),
        float(meta.get("bitterness", 0.5)),
        float(cosine_scores.get(coffee_name, 0.0)),
    ]


def _run_ablation(agent: BrewBuddyAgent) -> pd.DataFrame:
    df = _history_eval_frame(agent)
    if df.empty or len(df) < 8:
        return pd.DataFrame()

    train_x = []
    train_y = []
    for _, r in df.iterrows():
        coffee = str(r["coffee"])
        cos = r["cosine_scores"] if isinstance(r["cosine_scores"], dict) else {}
        train_x.append(_feature_vector(coffee, r, agent.coffee_items, cos))
        train_y.append(float(r["rating"]))
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(np.array(train_x), np.array(train_y))

    rng = random.Random(42)
    means = {}
    counts = {}
    rows = []
    for _, r in df.iterrows():
        cands = _to_list_safe(r["candidates"])
        if not cands:
            cands = list(agent.coffee_items.keys())[:8]
        cos_scores = r["cosine_scores"] if isinstance(r["cosine_scores"], dict) else {}
        pred = {}
        for c in cands:
            fv = np.array(_feature_vector(c, r, agent.coffee_items, cos_scores)).reshape(1, -1)
            pred[c] = float(model.predict(fv)[0])

        random_pick = rng.choice(cands)
        cosine_pick = max(cands, key=lambda c: float(cos_scores.get(c, 0.0)))
        content_pick = max(cands, key=lambda c: pred.get(c, -1e9))
        bandit_pick = max(
            cands,
            key=lambda c: (means.get(c, 0.0), -counts.get(c, 0)),
        )
        logged_pick = str(r["coffee"])

        rows.append(
            {
                "policy": "Hybrid (logged)",
                "predicted_reward": pred.get(logged_pick, 0.0),
                "observed_reward": float(r["rating"]),
            }
        )
        rows.append({"policy": "Cosine-only", "predicted_reward": pred.get(cosine_pick, 0.0), "observed_reward": np.nan})
        rows.append({"policy": "Content-only", "predicted_reward": pred.get(content_pick, 0.0), "observed_reward": np.nan})
        rows.append({"policy": "Bandit-mean", "predicted_reward": pred.get(bandit_pick, 0.0), "observed_reward": np.nan})
        rows.append({"policy": "Random", "predicted_reward": pred.get(random_pick, 0.0), "observed_reward": np.nan})

        # Update running bandit stats from the real logged outcome
        means[logged_pick] = (
            means.get(logged_pick, 0.0) * counts.get(logged_pick, 0) + float(r["rating"])
        ) / (counts.get(logged_pick, 0) + 1)
        counts[logged_pick] = counts.get(logged_pick, 0) + 1

    out = pd.DataFrame(rows)
    summary = (
        out.groupby("policy")
        .agg(
            predicted_reward=("predicted_reward", "mean"),
            observed_reward=("observed_reward", "mean"),
            n=("predicted_reward", "size"),
        )
        .reset_index()
    )
    return summary.sort_values("predicted_reward", ascending=False)


# Inject global styles (Outfit for UI, Fraunces for display)
st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,500&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
    :root {{
        --bb-bg: #070708;
        --bb-surface: rgba(18, 18, 24, 0.75);
        --bb-surface2: rgba(28, 28, 36, 0.9);
        --bb-border: rgba(212, 166, 116, 0.18);
        --bb-glow: rgba(212, 166, 116, 0.12);
        --bb-text: #ece8e2;
        --bb-text-muted: #8a8680;
        --bb-accent: {ACCENT};
        --bb-accent-deep: {ACCENT2};
        --bb-radius: 16px;
        --bb-radius-sm: 10px;
    }}
    .stApp {{
        background: var(--bb-bg) !important;
        background-image:
            radial-gradient(ellipse 120% 80% at 0% 0%, rgba(184, 107, 58, 0.14) 0%, transparent 50%),
            radial-gradient(ellipse 100% 60% at 100% 0%, rgba(90, 70, 120, 0.1) 0%, transparent 45%),
            radial-gradient(ellipse 60% 40% at 50% 100%, rgba(30, 28, 32, 1) 0%, #070708 100%) !important;
    }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header[data-testid="stHeader"] {{
        background: rgba(7,7,8,0.6) !important;
        backdrop-filter: blur(10px) !important;
    }}
    .block-container {{
        padding-top: 1.25rem;
        max-width: 1200px;
    }}
    /* Typography */
    h1, h2, h3, .stMarkdown p, .stText {{ font-family: 'Outfit', system-ui, sans-serif !important; color: var(--bb-text) !important; }}
    p {{ color: var(--bb-text-muted) !important; line-height: 1.6; }}
    .bb-display {{
        font-family: 'Fraunces', Georgia, serif !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }}
    /* Hero */
    .bb-hero {{
        text-align: center;
        max-width: 760px;
        margin: 0 auto 1.75rem auto;
        padding: 0.95rem 1.2rem 1.35rem 1.2rem;
        border: 1px solid rgba(212, 166, 116, 0.18);
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(19, 16, 23, 0.72) 0%, rgba(10, 10, 14, 0.55) 100%);
        box-shadow: 0 0 0 1px rgba(212, 166, 116, 0.06), 0 24px 48px -30px rgba(0, 0, 0, 0.7);
    }}
    .bb-hero-title {{
        font-family: 'Fraunces', Georgia, serif;
        font-size: clamp(2.55rem, 5vw, 3.3rem);
        font-weight: 700;
        background: linear-gradient(120deg, #f0e4d4 0%, {ACCENT} 45%, #8b5a3a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.45rem 0;
        text-shadow: 0 6px 20px rgba(212, 166, 116, 0.2);
    }}
    .bb-hero-sub {{
        font-family: 'Outfit', sans-serif;
        font-size: 1.02rem;
        font-weight: 400;
        color: var(--bb-text-muted) !important;
        max-width: 36rem;
        margin: 0.2rem auto 0 auto;
        text-align: center !important;
        display: block;
        width: 100%;
    }}
    .bb-pill-row {{
        display: flex; flex-wrap: wrap; justify-content: center; gap: 0.5rem; margin-top: 1.1rem;
    }}
    .bb-pill {{
        font-family: 'Outfit', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {ACCENT};
        border: 1px solid var(--bb-border);
        background: var(--bb-surface);
        padding: 0.4rem 0.75rem;
        border-radius: 999px;
    }}
    /* Glass cards */
    .bb-glass {{
        background: var(--bb-surface);
        border: 1px solid var(--bb-border);
        border-radius: var(--bb-radius);
        padding: 1.35rem 1.5rem;
        box-shadow: 0 0 0 1px var(--bb-glow), 0 24px 48px -24px rgba(0,0,0,0.5);
    }}
    .bb-glass-tight {{ padding: 1rem 1.15rem; border-radius: var(--bb-radius-sm); }}
    .bb-section-label {{
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: {ACCENT};
        margin-bottom: 0.5rem;
        font-family: 'Outfit', sans-serif;
    }}
    .bb-ctx-prel {{
        font-size: 0.78rem; color: var(--bb-text-muted) !important;
        line-height: 1.5;
        word-break: break-word;
        max-height: 3.2em; overflow: hidden;
    }}
    .bb-ml-badge {{
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        color: #1a1816;
        background: linear-gradient(120deg, {ACCENT} 0%, #c9a07a 100%);
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        text-transform: capitalize;
    }}
    /* Product card */
    .bb-drink-name {{
        font-family: 'Fraunces', Georgia, serif;
        font-size: 2rem;
        text-align: center;
        color: #f2ebe3 !important;
        margin: 0.5rem 0 0.35rem 0;
    }}
    .bb-drink-desc {{
        text-align: center;
        font-size: 1.02rem;
        color: var(--bb-text-muted) !important;
    }}
    .bb-rec-card {{
        background: linear-gradient(165deg, rgba(28, 22, 17, 0.92) 0%, rgba(17, 14, 20, 0.92) 100%);
        border: 1px solid rgba(212, 166, 116, 0.28);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        height: 100%;
    }}
    .bb-rec-title {{
        color: #f0e4d4 !important;
        font-size: 1.02rem;
        font-weight: 600;
        margin: 0 0 0.4rem 0;
    }}
    .bb-rec-meta {{
        color: #a89f95 !important;
        font-size: 0.78rem;
        margin: 0;
    }}
    .bb-chip {{
        display: inline-block;
        font-size: 0.68rem;
        padding: 0.2rem 0.46rem;
        border-radius: 999px;
        color: #f6e9d7;
        background: rgba(212, 166, 116, 0.2);
        border: 1px solid rgba(212, 166, 116, 0.35);
        margin-right: 0.25rem;
    }}
    .bb-kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.6rem;
        margin: 0.2rem 0 1rem 0;
    }}
    .bb-kpi {{
        background: linear-gradient(165deg, rgba(20, 18, 24, 0.92) 0%, rgba(14, 13, 18, 0.92) 100%);
        border: 1px solid rgba(212, 166, 116, 0.25);
        border-radius: 10px;
        padding: 0.6rem 0.7rem;
    }}
    .bb-kpi-label {{
        font-size: 0.62rem;
        color: var(--bb-text-muted) !important;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 0;
    }}
    .bb-kpi-value {{
        font-size: 1.05rem;
        color: #f2e2cf !important;
        font-family: 'Fraunces', serif;
        margin: 0.18rem 0 0 0;
    }}
    @media (max-width: 980px) {{
        .bb-kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    .bb-kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.6rem;
        margin: 0.2rem 0 1rem 0;
    }}
    .bb-kpi {{
        background: linear-gradient(165deg, rgba(20, 18, 24, 0.92) 0%, rgba(14, 13, 18, 0.92) 100%);
        border: 1px solid rgba(212, 166, 116, 0.25);
        border-radius: 10px;
        padding: 0.6rem 0.7rem;
    }}
    .bb-kpi-label {{
        font-size: 0.62rem;
        color: var(--bb-text-muted) !important;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 0;
    }}
    .bb-kpi-value {{
        font-size: 1.05rem;
        color: #f2e2cf !important;
        font-family: 'Fraunces', serif;
        margin: 0.18rem 0 0 0;
    }}
    @media (max-width: 980px) {{
        .bb-kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    /* Stats mini */
    .bb-stat-num {{ font-size: 1.85rem; font-weight: 700; color: {ACCENT}; font-family: 'Fraunces', serif; }}
    .bb-stat-label {{ font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--bb-text-muted) !important; font-weight: 600; }}
    /* Sliders: track and thumb */
    [data-testid="stVerticalBlock"] .stSlider label span {{ color: var(--bb-text-muted) !important; font-size: 0.85rem; }}
    .stSlider [data-baseweb="slider"] [role="slider"] {{ background: linear-gradient(180deg, {ACCENT}, {ACCENT2}) !important; box-shadow: 0 0 0 2px #0a0a0b, 0 0 0 3px {ACCENT} !important; }}
    .stSlider [data-baseweb="slider"] div[data-testid] {{ background: var(--bb-surface2) !important; }}
    /* Select & inputs */
    [data-baseweb="select"] > div {{ background: var(--bb-surface2) !important; border-color: var(--bb-border) !important; border-radius: var(--bb-radius-sm) !important; }}
    [data-baseweb="select"] span {{ color: var(--bb-text) !important; }}
    [data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background: linear-gradient(180deg, rgba(37, 34, 47, 0.98) 0%, rgba(29, 27, 38, 0.98) 100%) !important;
        border-color: rgba(212, 166, 116, 0.36) !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] div {{
        color: #f5e9dc !important;
        font-weight: 600 !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="select"] svg {{
        color: rgba(212, 166, 116, 0.95) !important;
    }}
    /* Checkboxes */
    [data-baseweb="checkbox"] span {{ color: var(--bb-text) !important; font-size: 0.9rem; }}
    /* Primary button */
    .stButton>button[kind="primary"] {{
        background: linear-gradient(120deg, {ACCENT2} 0%, {ACCENT} 100%) !important;
        color: #0d0b09 !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: 0.02em;
        box-shadow: 0 4px 24px -4px rgba(180, 120, 80, 0.45) !important;
        transition: transform 0.15s ease, box-shadow 0.2s;
    }}
    .stButton>button[kind="primary"]:hover {{
        box-shadow: 0 6px 32px -2px rgba(200, 140, 90, 0.55) !important;
    }}
    .stButton>button[kind="secondary"] {{
        background: var(--bb-surface2) !important;
        color: var(--bb-text) !important;
        border: 1px solid var(--bb-border) !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
    }}
    /* Default buttons in columns */
    .stButton>button {{ border-radius: 12px !important; font-family: 'Outfit', sans-serif !important; }}
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0c0c12 0%, #11121a 100%) !important;
        border-right: 1px solid var(--bb-border) !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        width: 360px !important;
    }}
    [data-testid="stSidebar"] {{
        min-width: 360px !important;
        max-width: 360px !important;
    }}
    [data-testid="stSidebar"] [data-baseweb] {{ color: var(--bb-text) !important; }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p {{ color: var(--bb-text) !important; }}
    [data-testid="stSidebar"] .stExpander details {{
        background: linear-gradient(180deg, rgba(33, 30, 42, 0.86) 0%, rgba(22, 22, 30, 0.86) 100%) !important;
        border: 1px solid rgba(212, 166, 116, 0.22) !important;
        border-radius: var(--bb-radius-sm) !important;
        box-shadow: 0 0 0 1px rgba(212, 166, 116, 0.05), inset 0 1px 0 rgba(255,255,255,0.03) !important;
        transition: border-color 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;
    }}
    [data-testid="stSidebar"] .stExpander details:hover {{
        border-color: rgba(212, 166, 116, 0.4) !important;
        background: linear-gradient(180deg, rgba(38, 34, 48, 0.95) 0%, rgba(25, 23, 34, 0.95) 100%) !important;
        box-shadow: 0 8px 24px -18px rgba(212, 166, 116, 0.35), 0 0 0 1px rgba(212, 166, 116, 0.12) !important;
    }}
    [data-testid="stSidebar"] .stExpander summary {{
        border-radius: 9px !important;
        padding: 0.16rem 0.3rem !important;
        background: transparent !important;
    }}
    [data-testid="stSidebar"] .stExpander summary:hover {{
        background: rgba(212, 166, 116, 0.08) !important;
    }}
    [data-testid="stSidebar"] .stExpander summary p {{ font-size: 0.88rem; font-weight: 600; }}
    [data-testid="stSidebar"] .stExpander summary svg {{
        color: rgba(212, 166, 116, 0.9) !important;
    }}
    /* Sidebar button tuning (avoid harsh hover flash) */
    [data-testid="stSidebar"] .stButton>button {{
        height: 2.35rem !important;
        min-height: 2.35rem !important;
        max-height: 2.35rem !important;
        width: 100% !important;
        min-width: 100% !important;
        padding: 0 !important;
        line-height: 1 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-sizing: border-box !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="stSidebar"] .stButton>button[kind="secondary"] {{
        background: rgba(40, 39, 52, 0.95) !important;
        border: 1px solid rgba(212, 166, 116, 0.26) !important;
        color: #e8e1d8 !important;
    }}
    [data-testid="stSidebar"] .stButton>button[kind="secondary"]:hover {{
        background: rgba(52, 50, 66, 0.98) !important;
        border-color: rgba(212, 166, 116, 0.45) !important;
        color: #f6ede2 !important;
        box-shadow: 0 10px 24px -16px rgba(212, 166, 116, 0.42) !important;
    }}
    [data-testid="stSidebar"] .stButton>button[kind="primary"] {{
        background: linear-gradient(120deg, #bb7a46 0%, #d8aa79 100%) !important;
        color: #16110c !important;
    }}
    [data-testid="stSidebar"] .stButton>button[kind="primary"]:hover {{
        background: linear-gradient(120deg, #c98650 0%, #e0b789 100%) !important;
        box-shadow: 0 12px 24px -14px rgba(217, 157, 101, 0.6) !important;
        transform: translateY(-1px);
    }}
    .bb-sidebar-kicker {{ font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.2em; color: {ACCENT}; font-weight: 600; margin-bottom: 0.2rem; }}
    .bb-sidebar-title {{ font-family: 'Fraunces', serif; font-size: 1.35rem; color: #f0e8e0; margin: 0 0 1.25rem 0; font-weight: 600; }}
    /* Tabs (analytics) */
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; background: transparent; border-bottom: 1px solid var(--bb-border) !important; padding-bottom: 4px; }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0 !important;
        color: var(--bb-text-muted) !important;
        font-weight: 500; font-size: 0.88rem;
    }}
    .stTabs [aria-selected="true"] p {{ color: {ACCENT} !important; font-weight: 600; }}
    .stTabs [data-baseweb="tab-highlight"] {{ background: linear-gradient(90deg, {ACCENT2}, {ACCENT}) !important; }}
    /* Streamlit status boxes */
    [data-testid="stSuccess"] {{ background: rgba(100, 180, 100, 0.12) !important; border: 1px solid rgba(120, 200, 120, 0.3) !important; border-radius: 12px !important; color: #c8e8c8 !important; }}
    [data-testid="stInfo"] {{ background: var(--bb-surface) !important; border: 1px solid var(--bb-border) !important; border-radius: 12px !important; color: var(--bb-text) !important; }}
    [data-testid="stWarning"] {{ background: rgba(255, 170, 70, 0.12) !important; border: 1px solid rgba(255, 170, 70, 0.35) !important; border-radius: 12px !important; color: #f1d2a7 !important; }}
    .bb-footer {{
        text-align: center; padding: 2rem 1rem; color: var(--bb-text-muted) !important;
        font-size: 0.78rem; letter-spacing: 0.04em;
    }}
    [data-testid="stExpander"] details summary {{ color: {ACCENT} !important; }}
    [data-testid="stExpanderDetails"] p,
    [data-testid="stExpanderDetails"] li,
    [data-testid="stExpanderDetails"] span {{
        color: #f4f4f6 !important;
    }}
    /* Metric streamlit */
    [data-testid="stMetricValue"] {{ color: {ACCENT} !important; font-family: 'Fraunces', serif !important; font-size: 1.5rem; }}
    [data-testid="stMetricLabel"] > div {{ color: var(--bb-text-muted) !important; text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.7rem; }}
    /* Dataframe */
    [data-testid="stDataFrame"] {{ border: 1px solid var(--bb-border) !important; border-radius: 12px !important; overflow: hidden; }}
    /* Divider */
    hr {{ border-color: var(--bb-border) !important; margin: 2rem 0 !important; }}
    .bb-sec-h {{ font-family: 'Fraunces', serif; font-size: 1.4rem; color: #e8e3dd !important; margin: 0 0 0.4rem 0; font-weight: 600; }}
    [data-testid="stImage"] img {{
        border-radius: 14px !important;
        box-shadow: 0 12px 40px -12px rgba(0,0,0,0.6), 0 0 0 1px rgba(212,166,116,0.15) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

init_db()
if "user_profile" not in st.session_state:
    st.session_state.user_profile = get_user_profile()

if "agent" not in st.session_state:
    st.session_state.agent = BrewBuddyAgent(
        coffees=None,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.3,
        use_context=True,
        use_subjective=True,
        use_hybrid=True,
        strategy="qlearning",
    )
    if os.path.exists("agent_state.json"):
        st.session_state.agent.load_state("agent_state.json")

if "background_worker" not in st.session_state:
    st.session_state.background_worker = BackgroundWorker(agent=st.session_state.agent, tick_interval=2.0)
    st.session_state.background_worker.start()

if "current_recommendation" not in st.session_state:
    st.session_state.current_recommendation = None
if "last_rating" not in st.session_state:
    st.session_state.last_rating = None
if "worker_status" not in st.session_state:
    st.session_state.worker_status = None
if "last_pending_recommendation" not in st.session_state:
    st.session_state.last_pending_recommendation = None

COFFEE_DESCRIPTIONS = {
    "Espresso": "Intense, concentrated, and unapologetically bold.",
    "Cappuccino": "Equal parts espresso, steamed milk, and airy foam.",
    "Latte": "Silky steamed milk over a base of smooth espresso.",
    "Americano": "Espresso opened up with hot water. Clean and direct.",
    "Mocha": "A gentle bridge between coffee and dark chocolate.",
    "Macchiato": "Espresso “marked” with a touch of foamed milk.",
    "Flat White": "Ristretto and velvety microfoam, compact and strong.",
    "Cortado": "Espresso and warm milk, cut in balance.",
    "Cold Brew": "Slow-steeped, smooth, and naturally low in acidity.",
    "Iced Coffee": "Chilled, refreshing, and ready to go.",
    "Frappuccino": "Blended, cool, and indulgent.",
    "Decaf": "The ritual without the buzz.",
}


def get_coffee_image_path(coffee_name: str):
    normalized = coffee_name.lower()
    image_mapping = {
        "espresso": "espresso.jpg",
        "cappuccino": "cappuccino.jpg",
        "latte": "latte.webp",
        "americano": "americano.jpg",
        "mocha": "mocha.png",
        "macchiato": "macchiato.jpg",
        "flat white": "flat white.jpg",
        "cortado": "cortado.webp",
        "cold brew": "cold brew.jpg",
        "iced coffee": "iced coffee.jpg",
        "frappuccino": "frappuccino.jpg",
        "decaf": "decaf.webp",
    }
    image_file = image_mapping.get(normalized)
    if image_file and os.path.exists(f"images/{image_file}"):
        return f"images/{image_file}"
    # Fallback for imported dataset coffees (no exact local asset yet)
    fallback_candidates = [
        "images/coffee1.png",
        "images/espresso.jpg",
        "images/americano.jpg",
    ]
    for fp in fallback_candidates:
        if os.path.exists(fp):
            return fp
    return None


# —— Sidebar (grouped) ——
with st.sidebar:
    st.markdown('<p class="bb-sidebar-kicker">Brewbuddy</p>', unsafe_allow_html=True)
    st.markdown('<p class="bb-sidebar-title">Control room</p>', unsafe_allow_html=True)

    with st.expander("Learning engine", expanded=True):
        strategy = st.selectbox(
            "Policy",
            ["qlearning", "thompson", "ucb"],
            index=0,
            help="Q-learning: value iteration over state–action. Thompson/UCB: multi-armed bandit.",
            label_visibility="visible",
        )
        if st.session_state.agent.strategy != strategy:
            st.session_state.agent.strategy = strategy
            if strategy == "thompson":
                st.session_state.agent.alpha = {c: 1.0 for c in st.session_state.agent.coffees}
                st.session_state.agent.beta = {c: 1.0 for c in st.session_state.agent.coffees}
            elif strategy == "ucb":
                st.session_state.agent.action_counts = {c: 0 for c in st.session_state.agent.coffees}
                st.session_state.agent.action_rewards = {c: [] for c in st.session_state.agent.coffees}
                st.session_state.agent.total_pulls = 0
        st.caption("Hyperparameters")
        learning_rate = st.slider("Learning rate (α)", 0.01, 0.5, 0.1, 0.01, label_visibility="visible")
        discount_factor = st.slider("Discount (γ)", 0.1, 0.99, 0.9, 0.01, label_visibility="visible")
        epsilon = st.slider("Exploration (ε)", 0.0, 1.0, 0.3, 0.05, label_visibility="visible")
        st.session_state.agent.learning_rate = learning_rate
        st.session_state.agent.discount_factor = discount_factor
        st.session_state.agent.epsilon = epsilon

    with st.expander("Environment", expanded=True):
        use_context = st.toggle("Context-aware state keys", value=True)
        st.session_state.agent.use_context = use_context
        if use_context:
            c1, c2 = st.columns(2)
            with c1:
                time_of_day = st.selectbox(
                    "Time",
                    [None, "morning", "afternoon", "evening", "night"],
                    index=0,
                    format_func=lambda t: t if t is not None else "-",
                )
            with c2:
                weather = st.selectbox("Sky", [None, "sunny", "rainy", "cloudy", "cold", "hot"], index=0, format_func=lambda w: w or "—")
            temperature = st.slider("Temperature (°C)", 0, 40, 20, 1)
        else:
            time_of_day, weather, temperature = None, None, None

    with st.expander("How you feel", expanded=True):
        use_subjective = st.toggle("Include internal state in features", value=True)
        st.session_state.agent.use_subjective = use_subjective
        f1, f2 = st.columns(2)
        with f1:
            sleep_h = st.slider("Sleep (h)", 0.0, 12.0, 7.0, 0.5, help="Last 24h")
        with f2:
            fatigue = st.slider("Fatigue", 1, 10, 5, 1, help="1 = fresh, 10 = drained")
        lactose = st.checkbox("Lactose intolerant (prefer low dairy load)", value=False)
        social = st.radio("Social battery", ["Full", "Empty"], horizontal=True, help="Affects need vector (comfort).")

    with st.expander("Taste profile", expanded=False):
        p_strong = st.slider(
            "Likes strong / bold caffeine",
            0.0,
            1.0,
            st.session_state.user_profile.get("pref_strong_caffeine", 0.5),
            0.05,
        )
        p_lf = st.toggle("Prefers lactose-free / no dairy in drinks", value=bool(st.session_state.user_profile.get("pref_lactose_free", 0)))
        p_bitter = st.slider("Likes mild / low bitterness", 0.0, 1.0, st.session_state.user_profile.get("pref_low_bitterness", 0.5), 0.05)
        c_save, _ = st.columns([1, 0.1])
        with c_save:
            if st.button("Save profile to SQLite", use_container_width=True, type="primary"):
                save_user_profile(p_strong, 1 if p_lf else 0, p_bitter)
                st.session_state.user_profile = get_user_profile()
                st.success("Profile saved")
        st.session_state.user_profile = {
            "pref_strong_caffeine": p_strong,
            "pref_lactose_free": 1 if p_lf else 0,
            "pref_low_bitterness": p_bitter,
        }

    with st.expander("Hybrid model", expanded=False):
        use_hybrid = st.toggle("Classify state → cosine shortlist → policy", value=True, help="Matches `state_category` in the database.")
        st.session_state.agent.use_hybrid = use_hybrid

    with st.expander("Competition data boost", expanded=False):
        st.caption("Import curated rows from the new datasets into SQLite and refresh the active catalog.")
        rows_per = st.slider("Rows per dataset", 100, 1200, 350, 50)
        if st.button("Import datasets into catalog", use_container_width=True, type="primary"):
            result = import_dataset_rows(limit_per_dataset=rows_per)
            st.session_state.agent.reload_catalog_from_db()
            st.success(
                f"Imported successfully: +{result['inserted']} new, {result['updated']} updated "
                f"(processed {result['seen']} rows)."
            )
        st.caption(f"Current catalog size: **{len(st.session_state.agent.coffees)}** drinks")

    st.divider()
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Save state", use_container_width=True, type="primary", help="agent_state.json"):
            st.session_state.agent.save_state()
            st.toast("Agent state saved")
    with b2:
        if st.button("Reset", use_container_width=True, type="primary"):
            if "background_worker" in st.session_state:
                st.session_state.background_worker.stop()
            st.session_state.agent = BrewBuddyAgent(
                coffees=None,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                use_context=use_context,
                use_subjective=use_subjective,
                use_hybrid=use_hybrid,
                strategy=strategy,
            )
            st.session_state.background_worker = BackgroundWorker(
                agent=st.session_state.agent,
                tick_interval=2.0,
            )
            st.session_state.background_worker.start()
            st.session_state.current_recommendation = None
            st.rerun()

# —— Main hero ——
st.markdown(
    """
<div class="bb-hero">
    <div class="bb-hero-title">BrewBuddy</div>
    <p class="bb-hero-sub"></p>
    <div class="bb-pill-row">
        <span class="bb-pill">Classification</span>
        <span class="bb-pill">Content match</span>
        <span class="bb-pill">Bandit / Q-learning</span>
    </div>
</div>""",
    unsafe_allow_html=True,
)

# —— Body ——
main, aside = st.columns([1.55, 0.9])

with main:
    st.markdown(
        """
        <p class="bb-section-label">Intelligence</p>
        <p class="bb-sec-h" style="margin:0 0 1rem 0;">Get your next recommendation</p>
        """,
        unsafe_allow_html=True,
    )

    subjective_payload = {
        "sleep_hours": float(sleep_h) if use_subjective else 7.0,
        "fatigue": int(fatigue) if use_subjective else 5,
        "lactose_intolerance": bool(lactose) if use_subjective else False,
        "social_battery": str(social) if use_subjective else "Full",
    }
    profile = st.session_state.user_profile
    st.session_state.agent.sense(
        time_of_day=time_of_day if use_context else None,
        weather=weather if use_context else None,
        temperature=temperature if use_context else None,
        subjective=subjective_payload if use_subjective else None,
        user_profile=profile,
    )
    ag = st.session_state.agent
    cand = ", ".join(ag._candidate_coffees[:3]) if use_hybrid and ag._candidate_coffees else ""
    ctx_short = (ag.current_context or "—")[:200]
    st.markdown(
        f"""
        <div class="bb-glass">
            <div class="bb-section-label">Live context</div>
            <p class="bb-ctx-prel">{ctx_short}{"…" if (ag.current_context and len(ag.current_context) > 200) else ""}</p>
            <p style="margin:0.75rem 0 0 0; display:flex; align-items:center; gap:0.6rem; flex-wrap:wrap;">
                <span class="bb-ml-badge">{ag.current_ml_state.replace("_", " ")}</span>
                <span style="color:#8a8680;font-size:0.8rem;">Top matches: {cand or "—"}</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    with st.expander("Engineering: need vector & cosine scores", expanded=False):
        st.caption("Four dimensions: stimulation, comfort, dairy concern, mildness. Compares to `coffee_items` in SQLite.")
        st.json({"need": ag.last_need_vector, "cosine": ag.last_cosine_scores or {}})

    if st.button("Request recommendation", type="primary", use_container_width=True, key="btn_get"):
        st.session_state.agent.add_context_request(
            time_of_day=time_of_day if use_context else None,
            weather=weather if use_context else None,
            temperature=temperature if use_context else None,
            subjective=subjective_payload if use_subjective else None,
            user_profile=profile,
        )
        st.info("Queued. The worker will pick this up (respects cooldown & pending rating).")
        st.rerun()

    current_pending = st.session_state.agent.pending_recommendation
    if current_pending != st.session_state.last_pending_recommendation:
        st.session_state.last_pending_recommendation = current_pending
        if current_pending is not None:
            st.session_state.current_recommendation = current_pending
            st.session_state.last_rating = None

    display_coffee: Optional[str] = None
    if current_pending is not None:
        display_coffee = current_pending
        if st.session_state.current_recommendation != display_coffee:
            st.session_state.current_recommendation = display_coffee
            st.session_state.last_rating = None
    else:
        latest_result = st.session_state.background_worker.get_latest_result()
        if latest_result is not None and not isinstance(latest_result, NoWork):
            if st.session_state.current_recommendation != latest_result:
                st.session_state.current_recommendation = latest_result
                st.session_state.last_rating = None
            display_coffee = latest_result
        elif isinstance(latest_result, NoWork):
            if st.session_state.current_recommendation is None:
                st.caption(f"⏳ {latest_result.reason}")
            display_coffee = st.session_state.current_recommendation
        else:
            display_coffee = st.session_state.current_recommendation
    st.markdown("")

    if display_coffee:
        coffee_name = display_coffee
        coffee_image_path = get_coffee_image_path(coffee_name)
        coffee_meta = ag.coffee_items.get(coffee_name, {})
        caffeine_level = float(coffee_meta.get("caffeine_level", 0.5))
        bitterness = float(coffee_meta.get("bitterness", 0.5))
        dairy_load = float(coffee_meta.get("dairy_load", 0.0))
        ml_state = ag.current_ml_state.replace("_", " ").title()
        st.markdown(
            """
        <div class="bb-glass" style="padding-bottom:0.5rem">
        <div class="bb-section-label" style="margin-top:0">This round</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        img_c, text_c = st.columns([0.95, 1.05])
        with img_c:
            if coffee_image_path and os.path.exists(coffee_image_path):
                st.image(Image.open(coffee_image_path), use_container_width=True, output_format="auto")
        with text_c:
            st.markdown(
                f"""
            <p class="bb-drink-name">{coffee_name}</p>
            <p class="bb-drink-desc">{COFFEE_DESCRIPTIONS.get(coffee_name, "A delicious option.")}</p>
            <p style="text-align:center; margin-top:0.35rem;">
                <span class="bb-chip">State: {ml_state}</span>
                <span class="bb-chip">Caffeine {caffeine_level:.2f}</span>
                <span class="bb-chip">Bitterness {bitterness:.2f}</span>
                <span class="bb-chip">Dairy {dairy_load:.2f}</span>
            </p>
            """,
                unsafe_allow_html=True,
            )
            top_alts = [c for c in ag._candidate_coffees if c != coffee_name][:3]
            if top_alts:
                st.markdown("**Alternative picks from current shortlist**")
                alt_cols = st.columns(len(top_alts))
                for i, alt in enumerate(top_alts):
                    alt_meta = ag.coffee_items.get(alt, {})
                    cos = ag.last_cosine_scores.get(alt, 0.0)
                    with alt_cols[i]:
                        st.markdown(
                            f"""
                            <div class="bb-rec-card">
                                <p class="bb-rec-title">{alt}</p>
                                <p class="bb-rec-meta">Cosine score: {cos:.3f}</p>
                                <p class="bb-rec-meta">Caffeine: {float(alt_meta.get("caffeine_level", 0.0)):.2f}</p>
                                <p class="bb-rec-meta">Dairy load: {float(alt_meta.get("dairy_load", 0.0)):.2f}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            with st.expander("Why this recommendation? (explainability narrative)", expanded=True):
                for bullet in ag.recommendation_narrative(coffee_name):
                    st.markdown(f"- {bullet}")
        st.divider()
        st.markdown("**How was it?**")
        rating = st.select_slider(
            "Tap to set rating",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: f"{'⭐' * x}  {x}/5",
            key="bb_rating",
        )
        r1, r2 = st.columns(2)
        with r1:
            if st.button("Submit & learn", type="primary", use_container_width=True, key="btn_learn"):
                st.session_state.agent.learn(coffee_name, int(rating))
                st.session_state.last_rating = rating
                st.session_state.agent.save_state()
                st.session_state.current_recommendation = None
                st.session_state.last_pending_recommendation = None
                st.rerun()
        with r2:
            if st.button("Request another", type="secondary", use_container_width=True, key="btn_another"):
                st.session_state.agent.add_context_request(
                    time_of_day=time_of_day if use_context else None,
                    weather=weather if use_context else None,
                    temperature=temperature if use_context else None,
                    subjective=subjective_payload if use_subjective else None,
                    user_profile=profile,
                )
                st.session_state.current_recommendation = None
                st.rerun()

with aside:
    st.markdown('<p class="bb-section-label">At a glance</p>', unsafe_allow_html=True)
    wstat = st.session_state.background_worker.get_status()
    if wstat["running"]:
        st.caption(f"Autonomous ticks: **{wstat['tick_count']}**")
    stats = st.session_state.agent.get_statistics()
    st.markdown('<div class="bb-glass bb-glass-tight" style="margin-bottom:0.8rem;">', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.metric("Session interactions", f"{stats['total_interactions']}")
    with b:
        st.metric("Average rating", f"{stats['average_rating']:.2f}")
    c, d = st.columns(2)
    with c:
        st.metric("Catalog size", f"{stats.get('menu_size', len(st.session_state.agent.coffees))}")
    with d:
        st.metric("Shortlist", f"{stats.get('shortlist_size', len(st.session_state.agent._candidate_coffees))}")
    st.markdown("</div>", unsafe_allow_html=True)
    if stats["best_coffee"]:
        st.markdown(
            f"""
        <div class="bb-glass bb-glass-tight">
            <div class="bb-stat-label" style="margin-bottom:0.2rem;">Best so far</div>
            <div class="bb-stat-num" style="font-size:1.2rem; color:#e8e3dd;">{stats['best_coffee']}</div>
            <p style="margin:0.25rem 0 0 0; color:#7a7772 !important; font-size:0.85rem;">{stats['best_rating']}/5</p>
        </div>""",
            unsafe_allow_html=True,
        )
    st.markdown("")

# —— Analytics ——
st.markdown("---")
st.markdown(
    '<p class="bb-section-label" style="margin-top:0.5rem;">Analytics</p><h3 class="bb-sec-h" style="font-size:1.35rem; margin:0 0 1.25rem 0;">Learning & performance</h3>',
    unsafe_allow_html=True,
)

_stats = st.session_state.agent.get_statistics()
_catalog_count = len(st.session_state.agent.coffee_items)
_history = st.session_state.agent.interaction_history
_n_states = len({h.get("context") for h in _history if h.get("context")}) if _history else 0
_source_badge = "seed"
if _catalog_count:
    source_vals = {
        (m.get("source_ref") or "seed")
        for m in st.session_state.agent.coffee_items.values()
    }
    if "dataset" in source_vals and len(source_vals) > 1:
        _source_badge = "mixed"
    elif "dataset" in source_vals:
        _source_badge = "dataset"

st.markdown(
    f"""
    <div class="bb-kpi-grid">
        <div class="bb-kpi"><p class="bb-kpi-label">Policy</p><p class="bb-kpi-value">{st.session_state.agent.strategy.upper()}</p></div>
        <div class="bb-kpi"><p class="bb-kpi-label">Interactions</p><p class="bb-kpi-value">{_stats.get("total_interactions", 0)}</p></div>
        <div class="bb-kpi"><p class="bb-kpi-label">Mean Rating</p><p class="bb-kpi-value">{_stats.get("average_rating", 0.0):.2f}</p></div>
        <div class="bb-kpi"><p class="bb-kpi-label">Catalog Source</p><p class="bb-kpi-value">{_source_badge.title()}</p></div>
    </div>
    <p style="margin:0 0 0.85rem 0;">
        <span class="bb-chip">State coverage: {_n_states}</span>
        <span class="bb-chip">Catalog size: {_catalog_count}</span>
        <span class="bb-chip">Shortlist: {len(st.session_state.agent._candidate_coffees)}</span>
        <span class="bb-chip">Hybrid: {"on" if st.session_state.agent.use_hybrid else "off"}</span>
    </p>
    """,
    unsafe_allow_html=True,
)

t1, t2, t3, t4, t5, t6 = st.tabs(["Q-Table", "By coffee", "Curve", "Context", "Catalog", "Validation"])

with t1:
    st.caption("State-action value landscape (Q-learning). Higher values indicate stronger policy preference.")
    if st.session_state.agent.strategy == "qlearning":
        q_df = st.session_state.agent.get_q_table_df()
        if not q_df.empty:
            fig = px.imshow(
                q_df.T,
                labels=dict(x="State", y="Coffee", color="Q"),
                color_continuous_scale=[[0, "rgba(8,8,10,0.3)"], [0.5, "rgba(212,166,116,0.55)"], [1, ACCENT2]],
                aspect="auto",
            )
            _apply_plot_theme(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.dataframe(q_df.style.format("{:.2f}").background_gradient(cmap="copper", axis=None), use_container_width=True, height=280)
        else:
            st.info("Interact and rate a few times to build the Q-table.")
    else:
        st.info("Switch the policy to Q-learning in the sidebar to see this view.")

with t2:
    st.caption("Mean feedback by coffee and number of trials (sample size encoded by color).")
    s2 = st.session_state.agent.get_statistics()
    cstats = s2["coffee_stats"]
    if cstats:
        names, avgs, cnts = [], [], []
        for c, d in cstats.items():
            names.append(c)
            avgs.append(d["avg_rating"])
            cnts.append(d["count"])
        dfp = pd.DataFrame({"Coffee": names, "Avg": avgs, "N": cnts})
        fig = px.bar(
            dfp, x="Coffee", y="Avg", color="N", text="Avg",
            color_continuous_scale=[[0, "rgba(8,8,10,0.2)"], [0.5, "rgba(212,166,116,0.60)"], [1, ACCENT2]]
        )
        _apply_plot_theme(fig)
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_line_width=0, marker_cornerradius=4)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.dataframe(dfp.sort_values("Avg", ascending=False), use_container_width=True, height=200)
    else:
        st.info("Ratings will appear here.")

with t3:
    h = st.session_state.agent.interaction_history
    if h:
        dfb = pd.DataFrame(h)
        dfb["cumulative_avg"] = dfb["rating"].expanding().mean()
        dfb["i"] = range(1, len(dfb) + 1)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dfb["i"],
                y=dfb["rating"],
                mode="markers+lines",
                name="Each rating",
                line=dict(color="rgba(212,166,116,0.40)", width=1),
                marker=dict(size=7, color=ACCENT, line=dict(width=0.5, color=ACCENT2)),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dfb["i"],
                y=dfb["cumulative_avg"],
                name="Cumulative",
                line=dict(color=ACCENT, width=2.5, shape="spline"),
            )
        )
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font_size=10))
        _apply_plot_theme(fig)
        fig.update_layout(xaxis_title="Interaction index", yaxis_title="Rating")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.subheader("Recent")
        keys = [k for k in ("coffee", "rating", "context", "ml_state", "subjective", "timestamp") if any(k in r for r in h[-10:])]
        if not keys and h:
            keys = [k for k in list(h[0].keys()) if k in h[-1]]
        if not keys:
            keys = ["coffee", "rating", "timestamp"]
        cols = [c for c in keys if c in h[-1]]
        recent = pd.DataFrame(h[-10:])[[c for c in cols]] if h and cols else None
        if recent is not None and "timestamp" in recent.columns:
            recent = recent.copy()
            recent["timestamp"] = pd.to_datetime(recent["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(recent, use_container_width=True, height=200, hide_index=True)
    else:
        st.info("Your learning curve starts after the first rating.")

with t4:
    dfc = st.session_state.agent
    if dfc.use_context and dfc.interaction_history:
        dfx = pd.DataFrame(dfc.interaction_history)
        if "context" in dfx.columns:
            csg = (
                dfx.groupby("context", dropna=True)
                .agg(rm=("rating", "mean"), c=("rating", "size"), pop=("coffee", lambda s: s.mode().iloc[0] if not s.empty and len(s.mode()) else None))
                .reset_index()
            )
            if not csg.empty:
                csg = csg.rename(columns={"context": "Context", "rm": "Avg", "c": "N", "pop": "Mode"})
                top_rows = csg.sort_values(["Avg", "N"], ascending=[False, False]).head(3)
                top_str = " | ".join([f"{r.Context[:24]} ({r.Avg:.2f})" for r in top_rows.itertuples(index=False)])
                st.caption(f"Top-performing contexts: {top_str}")
                fig = px.bar(
                    csg, x="Context", y="Avg", color="N", text="Avg",
                    color_continuous_scale=[[0, "rgba(5,5,8,0.2)"], [0.5, "rgba(212,166,116,0.55)"], [1, ACCENT2]]
                )
                _apply_plot_theme(fig)
                fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_cornerradius=4)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.dataframe(csg, use_container_width=True, height=220)
            else:
                st.caption("No grouped rows yet.")
        else:
            st.caption("Older history had no `context` field; new runs log the full key.")
    else:
        st.info("Enable context and add interactions.")

with t5:
    st.caption("Catalog composition and source quality (seed vs imported datasets).")
    catalog_rows = []
    for n, meta in st.session_state.agent.coffee_items.items():
        catalog_rows.append(
            {
                "Coffee": n,
                "Category": meta.get("state_category"),
                "Source": meta.get("source_ref") or "seed",
                "Caffeine": float(meta.get("caffeine_level", 0)),
                "Bitterness": float(meta.get("bitterness", 0)),
                "Dairy": float(meta.get("dairy_load", 0)),
            }
        )
    if catalog_rows:
        cdf = pd.DataFrame(catalog_rows)
        source_counts = cdf.groupby("Source").size().reset_index(name="Count")
        fig = px.pie(source_counts, values="Count", names="Source", hole=0.45)
        _apply_plot_theme(fig)
        fig.update_traces(
            textinfo="percent+label",
            marker=dict(line=dict(color="rgba(255,255,255,0.08)", width=1)),
        )
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.dataframe(
            cdf.sort_values(["Source", "Coffee"]).reset_index(drop=True),
            use_container_width=True,
            height=280,
        )
    else:
        st.info("No catalog rows available.")

with t6:
    st.caption("Evaluation and ablation summary from logged interactions.")
    hist_df = _history_eval_frame(st.session_state.agent)
    if hist_df.empty or len(hist_df) < 5:
        st.info("Need at least 5 rated interactions to run validation and ablation.")
    else:
        by_state = (
            hist_df.groupby("ml_state")
            .agg(avg_rating=("rating", "mean"), n=("rating", "size"))
            .reset_index()
            .sort_values(["avg_rating", "n"], ascending=[False, False])
        )
        state_var = float(by_state["avg_rating"].var()) if len(by_state) > 1 else 0.0
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("States observed", len(by_state))
        with c2:
            st.metric("Between-state variance", f"{state_var:.3f}")
        with c3:
            st.metric("Validation samples", len(hist_df))

        fig = px.bar(
            by_state,
            x="ml_state",
            y="avg_rating",
            color="n",
            text="avg_rating",
            color_continuous_scale=[[0, "rgba(8,8,10,0.25)"], [0.5, "rgba(212,166,116,0.55)"], [1, ACCENT2]],
            title="Average reward by predicted state",
        )
        _apply_plot_theme(fig)
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.dataframe(by_state.rename(columns={"ml_state": "State", "avg_rating": "Avg Rating", "n": "Count"}), use_container_width=True)

        ablation = _run_ablation(st.session_state.agent)
        st.markdown("#### Ablation Comparison (Offline Counterfactual)")
        if ablation.empty:
            st.info("Not enough data for ablation yet (need at least ~8 interactions).")
        else:
            fig2 = px.bar(
                ablation,
                x="policy",
                y="predicted_reward",
                color="policy",
                text="predicted_reward",
                title="Estimated reward by policy variant",
            )
            _apply_plot_theme(fig2)
            fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            st.dataframe(
                ablation.rename(
                    columns={
                        "policy": "Policy",
                        "predicted_reward": "Estimated Reward",
                        "observed_reward": "Observed Reward (logged only)",
                        "n": "Samples",
                    }
                ),
                use_container_width=True,
            )

st.markdown(
    f"""
    <div class="bb-footer">
        BrewBuddy · state classification, cosine match, and RL policy &nbsp;|&nbsp;
        <code style="color:{PLOT_MUTED};">data/brewbuddy.db</code> &amp; <code style="color:{PLOT_MUTED};">agent_state.json</code>
    </div>
    """,
    unsafe_allow_html=True,
)
