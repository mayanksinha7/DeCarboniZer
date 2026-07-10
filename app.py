"""
Carbon Footprint Predictor — Streamlit Prototype
=================================================
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
from scipy import stats
import re

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Carbon Footprint Predictor",
    page_icon="🌿",
    layout="wide",
)

st.markdown(
    """
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

    /* ── Root Variables ── */
    :root {
        --forest:    #fbfdf9;
        --canopy:    #f0f7f1;
        --moss:      #dcf0e3;
        --fern:      #0f7a59;
        --leaf:      #138a64;
        --lime:      #3f9e2f;
        --amber:     #b8852a;
        --ember:     #c2461f;
        --bark:      #fdf3e7;
        --cream:     #16241c;
        --mist:      rgba(15,30,20,0.06);
        --mist2:     rgba(15,30,20,0.08);
        --glow:      0 0 40px rgba(15,122,89,0.18);
    }

    /* ── Global reset ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--forest) !important;
        color: var(--cream) !important;
    }

    /* ── App background with organic texture ── */
    .stApp {
        background:
            radial-gradient(ellipse 80% 50% at 20% 0%, rgba(15,122,89,0.10) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 100%, rgba(63,158,47,0.06) 0%, transparent 50%),
            radial-gradient(ellipse 100% 80% at 50% 50%, rgba(255,255,255,1) 0%, rgba(255,255,255,1) 100%);
        background-attachment: fixed;
    }

    /* ── Block container ── */
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }

    /* ── HERO HEADER ── */
    .hero-header {
        background: linear-gradient(135deg, rgba(255,255,255,0.97) 0%, rgba(220,240,227,0.95) 100%);
        border-bottom: 1px solid rgba(15,122,89,0.32);
        padding: 2.5rem 3rem 2rem;
        margin: -1rem -1rem 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        inset: 0;
        background:
            radial-gradient(circle at 10% 50%, rgba(15,122,89,0.14) 0%, transparent 40%),
            radial-gradient(circle at 90% 20%, rgba(63,158,47,0.10) 0%, transparent 40%);
        pointer-events: none;
    }
    .hero-eyebrow {
        font-family: 'DM Sans', sans-serif;
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: var(--fern);
        margin: 0 0 0.5rem;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 800;
        line-height: 1.1;
        margin: 0 0 0.75rem;
        background: linear-gradient(135deg, #16241c 30%, #3f9e2f 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub {
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        font-weight: 300;
        color: rgba(22,36,28,0.78);
        max-width: 560px;
        line-height: 1.6;
        margin: 0;
    }
    .hero-leaf {
        position: absolute;
        right: 3rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 7rem;
        opacity: 0.07;
        pointer-events: none;
        filter: blur(2px);
    }

    /* ── METRIC CARDS ── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, var(--canopy) 0%, var(--moss) 100%) !important;
        border: 1px solid rgba(15,122,89,0.25) !important;
        border-radius: 16px !important;
        padding: 1.25rem 1.5rem !important;
        box-shadow: var(--glow), inset 0 1px 0 rgba(15,30,20,0.05) !important;
        transition: border-color 0.3s, box-shadow 0.3s !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(15,122,89,0.50) !important;
        box-shadow: 0 0 50px rgba(15,122,89,0.28), inset 0 1px 0 rgba(15,30,20,0.05) !important;
    }
    [data-testid="stMetricLabel"] p {
        color: rgba(22,36,28,0.72) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Syne', sans-serif !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #16241c !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
    }

    /* ── SCORE DISPLAY ── */
    .score-green  { color: var(--fern);  font-family: 'Syne', sans-serif; font-size: 52px; font-weight: 800; line-height: 1; }
    .score-amber  { color: var(--amber); font-family: 'Syne', sans-serif; font-size: 52px; font-weight: 800; line-height: 1; }
    .score-red    { color: var(--ember); font-family: 'Syne', sans-serif; font-size: 52px; font-weight: 800; line-height: 1; }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--canopy) 0%, var(--bark) 100%) !important;
        border-right: 1px solid rgba(15,122,89,0.18) !important;
    }
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 200px;
        background: radial-gradient(ellipse at 50% -20%, rgba(15,122,89,0.18) 0%, transparent 70%);
        pointer-events: none;
    }
    [data-testid="stSidebarContent"] {
        padding-top: 1.5rem !important;
    }

    /* Sidebar header */
    [data-testid="stSidebar"] h2 {
        font-family: 'Syne', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        color: var(--leaf) !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: rgba(22,36,28,0.72) !important;
        font-size: 0.85rem !important;
    }

    /* ── SIDEBAR EXPANDERS ── */
    [data-testid="stSidebar"] .stExpander {
        border: 1px solid rgba(15,122,89,0.18) !important;
        border-radius: 12px !important;
        margin-bottom: 0.5rem !important;
        background: rgba(15,122,89,0.06) !important;
        overflow: hidden !important;
    }
    [data-testid="stSidebar"] .stExpander:hover {
        border-color: rgba(15,122,89,0.38) !important;
    }
    [data-testid="stSidebar"] .stExpander summary {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        color: rgba(22,36,28,0.92) !important;
        font-size: 0.9rem !important;
    }
    /* Fix expander summary (header button) white background */
    [data-testid="stSidebar"] .stExpander summary,
    [data-testid="stSidebar"] details summary,
    [data-testid="stSidebar"] [data-testid="stExpanderToggleIcon"],
    [data-testid="stSidebar"] details > summary {
        background: rgba(255,255,255,0.92) !important;
        background-color: rgba(255,255,255,0.92) !important;
        color: rgba(22,36,28,0.92) !important;
    }
    [data-testid="stSidebar"] details > summary:hover {
        background: rgba(15,122,89,0.14) !important;
    }

    /* ── FORM WIDGETS ── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: rgba(255,255,255,0.85) !important;
        border: 1px solid rgba(15,122,89,0.25) !important;
        border-radius: 10px !important;
        color: var(--cream) !important;
        transition: border-color 0.2s !important;
    }
    .stSelectbox > div > div:hover,
    .stMultiSelect > div > div:hover {
        border-color: rgba(15,122,89,0.55) !important;
    }
    /* Selected value text + dropdown arrow inside selects */
    .stSelectbox [data-baseweb="select"] *,
    .stMultiSelect [data-baseweb="select"] * {
        color: var(--cream) !important;
    }
    .stSelectbox [data-baseweb="select"] svg,
    .stMultiSelect [data-baseweb="select"] svg {
        fill: var(--cream) !important;
    }
    /* Dropdown menu popover (rendered in a portal) */
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="menu"] {
        background-color: var(--canopy) !important;
        border: 1px solid rgba(15,122,89,0.38) !important;
    }
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="menu"] li {
        color: var(--cream) !important;
        background-color: var(--canopy) !important;
    }
    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="menu"] li:hover {
        background-color: rgba(15,122,89,0.32) !important;
    }

    /* Widget labels (Body type, Sex, Diet type, sliders, etc.) */
    .stSelectbox label p,
    .stMultiSelect label p,
    .stSlider label p,
    .stTextInput label p,
    .stNumberInput label p,
    .stTextArea label p,
    .stRadio label p,
    .stCheckbox label p,
    div[data-testid="stWidgetLabel"] p {
        color: rgba(22,36,28,0.88) !important;
    }

    /* Text area (e.g. "Changes you made") */
    .stTextArea textarea {
        background-color: rgba(255,255,255,0.92) !important;
        color: var(--cream) !important;
        border: 1px solid rgba(15,122,89,0.25) !important;
    }
    .stTextArea textarea::placeholder {
        color: rgba(22,36,28,0.68) !important;
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] {
        padding: 4px 0 !important;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: var(--fern) !important;
        border: 2px solid var(--leaf) !important;
        box-shadow: 0 0 12px rgba(15,122,89,0.55) !important;
    }
    .stSlider div[data-testid="stSliderThumbValue"] {
        color: var(--leaf) !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
    }

    /* ── ALL BUTTONS — catch-all so nothing is invisible on dark bg ── */
    .stButton > button {
        background: rgba(15,122,89,0.12) !important;
        border: 1px solid rgba(19,138,100,0.45) !important;
        color: #138a64 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        border-radius: 10px !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: rgba(15,122,89,0.28) !important;
        border-color: var(--leaf) !important;
        color: #ffffff !important;
    }
    .stButton > button:focus:not(:active) {
        outline: 2px solid rgba(19,138,100,0.50) !important;
        outline-offset: 2px !important;
    }

    /* ── PRIMARY BUTTON ── */
    .stButton > button[kind="primary"],
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, var(--fern) 0%, var(--leaf) 100%) !important;
        color: #ffffff !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.04em !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.85rem 2rem !important;
        box-shadow: 0 4px 24px rgba(15,122,89,0.42), inset 0 1px 0 rgba(15,30,20,0.15) !important;
        transition: all 0.25s ease !important;
        text-transform: uppercase !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.12) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 32px rgba(15,122,89,0.60), inset 0 1px 0 rgba(15,30,20,0.15) !important;
    }

    /* Secondary button */
    .stButton > button[kind="secondary"] {
        background: rgba(15,122,89,0.10) !important;
        border: 1px solid rgba(19,138,100,0.55) !important;
        color: #138a64 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        border-radius: 10px !important;
        transition: all 0.2s !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(15,122,89,0.20) !important;
        border-color: var(--leaf) !important;
        color: #ffffff !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, rgba(15,122,89,0.28) 0%, rgba(19,138,100,0.18) 100%) !important;
        border: 1px solid rgba(19,138,100,0.55) !important;
        color: #138a64 !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
        border-radius: 12px !important;
        transition: all 0.25s !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(15,122,89,0.40) !important;
        border-color: var(--leaf) !important;
        color: #ffffff !important;
        box-shadow: 0 0 20px rgba(15,122,89,0.42) !important;
    }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--canopy) !important;
        border-radius: 14px !important;
        padding: 5px !important;
        gap: 4px !important;
        border: 1px solid rgba(15,122,89,0.18) !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 10px !important;
        color: rgba(22,36,28,0.70) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1.25rem !important;
        transition: all 0.2s !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--moss) 0%, rgba(15,122,89,0.25) 100%) !important;
        color: var(--leaf) !important;
        border: 1px solid rgba(15,122,89,0.38) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* ── ALERT / INFO / SUCCESS / WARNING BANNERS ── */
    .stAlert {
        border-radius: 12px !important;
        border: 1px solid !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    div[data-testid="stNotification"]:has(.st-ae) {   /* success */
        background: rgba(15,122,89,0.12) !important;
        border-color: rgba(15,122,89,0.42) !important;
    }
    div[data-testid="stNotification"]:has(.st-af) {   /* warning */
        background: rgba(184,133,42,0.10) !important;
        border-color: rgba(184,133,42,0.32) !important;
    }
    .stSuccess {
        background: rgba(15,122,89,0.12) !important;
        border-color: rgba(15,122,89,0.42) !important;
        color: rgba(22,36,28,0.95) !important;
        border-radius: 12px !important;
    }
    .stWarning {
        background: rgba(184,133,42,0.10) !important;
        border-color: rgba(184,133,42,0.32) !important;
        color: rgba(22,36,28,0.95) !important;
        border-radius: 12px !important;
    }
    .stInfo {
        background: rgba(15,122,89,0.09) !important;
        border-color: rgba(15,122,89,0.25) !important;
        color: rgba(22,36,28,0.88) !important;
        border-radius: 12px !important;
    }
    .stError {
        background: rgba(194,70,31,0.10) !important;
        border-color: rgba(194,70,31,0.30) !important;
        border-radius: 12px !important;
    }

    /* ── CONTAINERS / CARDS ── */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        border: 1px solid rgba(15,122,89,0.25) !important;
        border-radius: 16px !important;
        background: linear-gradient(135deg, rgba(255,255,255,0.90) 0%, rgba(220,240,227,0.60) 100%) !important;
        box-shadow: var(--glow) !important;
        transition: border-color 0.3s, box-shadow 0.3s !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div:hover {
        border-color: rgba(15,122,89,0.45) !important;
        box-shadow: 0 0 50px rgba(15,122,89,0.20) !important;
    }

    /* ── EXPANDERS (main area) ── */
    .stExpander {
        border: 1px solid rgba(15,122,89,0.20) !important;
        border-radius: 14px !important;
        background: rgba(255,255,255,0.75) !important;
    }
    /* Summary / header row */
    .stExpander > summary,
    .stExpander details > summary,
    details.stExpander > summary,
    [data-testid="stExpander"] > details > summary,
    [data-testid="stExpander"] summary {
        background: rgba(255,255,255,0.92) !important;
        background-color: rgba(255,255,255,0.92) !important;
        color: #16241c !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    /* Expanded content area — this is what was going white */
    .stExpander details[open],
    .stExpander details,
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] details[open] {
        background: rgba(255,255,255,0.85) !important;
        background-color: rgba(255,255,255,0.85) !important;
    }
    /* Inner content wrapper */
    .stExpander details > div,
    [data-testid="stExpander"] details > div {
        background: transparent !important;
        background-color: transparent !important;
    }
    .stExpander summary p,
    .stExpander details summary p,
    [data-testid="stExpander"] summary p {
        color: #16241c !important;
    }
    /* Dataframe inside expander — ensure no white bleed */
    .stExpander [data-testid="stDataFrame"] > div {
        background: var(--canopy) !important;
        color: #16241c !important;
    }

    /* ── DATAFRAMES ── */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(15,122,89,0.25) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] > div,
    [data-testid="stDataFrame"] [data-testid="glideDataEditor"],
    [data-testid="stDataFrame"] .dvn-scroller,
    [data-testid="stDataFrame"] canvas {
        background-color: #f0f7f1 !important;
        color-scheme: dark !important;
    }
    [data-testid="stDataFrame"] {
        --gdg-bg-cell: #f0f7f1;
        --gdg-bg-cell-medium: #e7f3ec;
        --gdg-bg-header: #fbfdf9;
        --gdg-bg-header-has-focus: #dcf0e3;
        --gdg-bg-header-hovered: #dcf0e3;
        --gdg-text-dark: #16241c;
        --gdg-text-medium: #16241c;
        --gdg-text-light: rgba(22,36,28,0.78);
        --gdg-text-header: #138a64;
        --gdg-text-header-selected: #ffffff;
        --gdg-border-color: rgba(15,122,89,0.25);
        --gdg-horizontal-border-color: rgba(15,122,89,0.18);
        --gdg-link-color: #138a64;
        --gdg-accent-color: #0f7a59;
        --gdg-accent-light: rgba(15,122,89,0.25);
        --gdg-bg-bubble: #dcf0e3;
        --gdg-bg-bubble-selected: #0f7a59;
        --gdg-bg-search-result: rgba(184,133,42,0.28);
    }

    /* ── ST.TABLE (Changes you made) ── */
    [data-testid="stTable"] table {
        background: var(--canopy) !important;
        border: 1px solid rgba(15,122,89,0.25) !important;
        border-radius: 12px !important;
    }
    [data-testid="stTable"] th,
    [data-testid="stTable"] td {
        color: var(--cream) !important;
        background: transparent !important;
        border-color: rgba(15,122,89,0.18) !important;
    }
    [data-testid="stTable"] thead th {
        color: var(--leaf) !important;
        font-family: 'DM Sans', sans-serif !important;
        text-transform: uppercase !important;
        font-size: 12px !important;
        letter-spacing: 0.05em !important;
    }
    [data-testid="stTable"] tbody tr:hover td {
        background: rgba(15,122,89,0.10) !important;
    }

    /* ── DIVIDER ── */
    hr {
        border-color: rgba(15,122,89,0.18) !important;
        margin: 1.5rem 0 !important;
    }

    /* ── SUBHEADERS ── */
    h2, h3 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        color: var(--cream) !important;
    }
    h2 { font-size: 1.5rem !important; }
    h3 { font-size: 1.15rem !important; }

    /* ── SPINNER ── */
    .stSpinner > div {
        border-color: var(--fern) var(--fern) transparent transparent !important;
    }

    /* ── SECTION LABEL (used inside tab for category labels) ── */
    .section-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--fern);
        margin: 0 0 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .section-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(15,122,89,0.38) 0%, transparent 100%);
    }

    /* ── ECO TIP CARDS ── */
    .eco-tip-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.97) 0%, rgba(255,255,255,0.90) 100%);
        border: 1px solid rgba(15,122,89,0.35);
        border-radius: 20px;
        padding: 1.5rem 1.6rem 1.35rem;
        margin-bottom: 0.85rem;
        transition: border-color 0.28s, transform 0.28s, box-shadow 0.28s;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 24px rgba(0,0,0,0.10), inset 0 1px 0 rgba(15,30,20,0.04);
    }
    .eco-tip-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--fern) 0%, var(--lime) 60%, transparent 100%);
        opacity: 0.75;
    }
    .eco-tip-card::after {
        content: '';
        position: absolute;
        bottom: -40px; right: -20px;
        width: 100px; height: 100px;
        background: radial-gradient(circle, rgba(15,122,89,0.10) 0%, transparent 70%);
        pointer-events: none;
    }
    .eco-tip-card:hover {
        border-color: rgba(19,138,100,0.55);
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(15,122,89,0.25), inset 0 1px 0 rgba(15,30,20,0.04);
    }
    .eco-tip-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.65rem;
    }
    .eco-tip-icon {
        font-size: 1.65rem;
        line-height: 1;
        flex-shrink: 0;
        filter: drop-shadow(0 0 6px rgba(15,122,89,0.45));
    }
    .eco-tip-title {
        font-family: 'Syne', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: var(--cream);
        margin: 0;
        letter-spacing: 0.01em;
    }
    .eco-tip-detail {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        font-weight: 300;
        color: rgba(22,36,28,0.78);
        line-height: 1.55;
        margin: 0 0 0.8rem;
    }
    .eco-tip-detail strong {
        color: var(--leaf);
        font-weight: 500;
    }
    .eco-tip-divider {
        height: 1px;
        background: linear-gradient(90deg, rgba(15,122,89,0.32) 0%, transparent 100%);
        margin: 0.6rem 0 0.75rem;
    }
    .eco-tip-tips-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--fern);
        margin-bottom: 0.5rem;
    }
    .eco-tip-list {
        list-style: none;
        padding: 0;
        margin: 0;
        display: flex;
        flex-direction: column;
        gap: 0.38rem;
    }
    .eco-tip-list li {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.82rem;
        font-weight: 400;
        color: rgba(22,36,28,0.83);
        line-height: 1.5;
        padding-left: 1.1rem;
        position: relative;
    }
    .eco-tip-list li::before {
        content: '▸';
        position: absolute;
        left: 0;
        color: var(--fern);
        font-size: 0.75rem;
        top: 0.05em;
    }

    /* ── PEER BANNER ── */
    .peer-banner-good {
        background: linear-gradient(135deg, rgba(15,122,89,0.18) 0%, rgba(63,158,47,0.10) 100%);
        border: 1px solid rgba(15,122,89,0.42);
        border-radius: 14px;
        padding: 1rem 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.95rem;
        color: rgba(22,36,28,0.95);
    }
    .peer-banner-warn {
        background: linear-gradient(135deg, rgba(184,133,42,0.12) 0%, rgba(194,70,31,0.06) 100%);
        border: 1px solid rgba(184,133,42,0.32);
        border-radius: 14px;
        padding: 1rem 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.95rem;
        color: rgba(22,36,28,0.95);
    }

    /* ── CAPTION / FOOTER ── */
    .stCaption, .stCaption p {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.78rem !important;
        color: rgba(22,36,28,0.60) !important;
        letter-spacing: 0.02em !important;
    }

    /* ── SPINNER TEXT ── */
    [data-testid="stSpinner"] p {
        color: var(--leaf) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ── MULTISELECT TAGS ── */
    [data-baseweb="tag"] {
        background: rgba(15,122,89,0.25) !important;
        border: 1px solid rgba(15,122,89,0.42) !important;
        color: var(--leaf) !important;
        border-radius: 6px !important;
    }

    /* ── Plotly chart containers ── */
    .js-plotly-plot .plotly, .plot-container {
        border-radius: 12px !important;
    }

    /* ── BASELINE DISPLAY BOX ── */
    .baseline-box {
        background: linear-gradient(135deg, rgba(255,255,255,0.97) 0%, rgba(15,122,89,0.10) 100%);
        border: 1px solid rgba(15,122,89,0.32);
        border-radius: 14px;
        padding: 1rem 1.5rem;
        font-family: 'Syne', sans-serif;
        font-size: 0.95rem;
        color: rgba(22,36,28,0.82);
        margin-bottom: 1rem;
    }
    .baseline-box span {
        color: var(--leaf);
        font-weight: 700;
    }

    /* ── WHAT-IF CATEGORY LABELS ── */
    .wi-category {
        font-family: 'Syne', sans-serif;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--fern);
        border-left: 3px solid var(--fern);
        padding-left: 0.6rem;
        margin-bottom: 0.75rem;
    }

</style>
""",
    unsafe_allow_html=True,
)


# ── Load artifacts ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    explainer = joblib.load("explainer.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    feature_names = joblib.load("feature_names.pkl")
    log_transformed = joblib.load("log_transformed.pkl")
    try:
        dropped_features = joblib.load("dropped_features.pkl")
    except FileNotFoundError:
        dropped_features = []
    try:
        all_train_preds = joblib.load("all_train_preds.pkl")
    except FileNotFoundError:
        all_train_preds = None
    try:
        ordinal_maps = joblib.load("ordinal_maps.pkl")
    except FileNotFoundError:
        ordinal_maps = {
            "Frequency of Traveling by Air": {
                "never": 0,
                "rarely": 1,
                "frequently": 2,
                "very frequently": 3,
            },
            "How Often Shower": {
                "less frequently": 0,
                "daily": 1,
                "more frequently": 2,
                "twice a day": 3,
            },
            "Waste Bag Size": {"small": 0, "medium": 1, "large": 2, "extra large": 3},
            "Body Type": {"underweight": 0, "normal": 1, "overweight": 2, "obese": 3},
            "Social Activity": {"never": 0, "sometimes": 1, "often": 2},
            "Energy efficiency": {"No": 0, "Sometimes": 1, "Yes": 2},
        }
    return (
        model,
        scaler,
        explainer,
        label_encoders,
        feature_names,
        log_transformed,
        dropped_features,
        all_train_preds,
        ordinal_maps,
    )


(
    model,
    scaler,
    explainer,
    label_encoders,
    feature_names,
    log_transformed,
    dropped_features,
    all_train_preds,
    ordinal_maps,
) = load_artifacts()


# ── Peer percentile ───────────────────────────────────────────────────────
def get_peer_percentile(emission_weekly, all_preds):
    if all_preds is None or len(all_preds) == 0:
        return None
    percentile = stats.percentileofscore(all_preds, emission_weekly, kind="rank")
    better_than = round(100 - percentile)
    return better_than


# ── preprocess_input ──────────────────────────────────────────────────────
def preprocess_input(inputs: dict) -> pd.DataFrame:
    row = {}

    ordinal_input_map = {
        "Body Type": inputs["body_type"],
        "How Often Shower": inputs["shower_freq"],
        "Frequency of Traveling by Air": inputs["air_travel"],
        "Waste Bag Size": inputs["waste_type"],
        "Social Activity": inputs["social_activity"],
        "Energy efficiency": inputs["energy_eff"],
    }
    for col, val in ordinal_input_map.items():
        mapping = ordinal_maps.get(col, {})
        row[col] = float(mapping.get(str(val).strip().lower(), 0))

    nominal_fields = {
        "Sex": inputs["sex"],
        "Diet": inputs["diet"],
        "Heating Energy Source": inputs["heating_source"],
        "Transport": inputs["transport"],
        "Vehicle Type": inputs["vehicle_type"],
    }
    for col, val in nominal_fields.items():
        if col in label_encoders:
            try:
                row[col] = int(label_encoders[col].transform([str(val)])[0])
            except ValueError:
                row[col] = 0
                st.warning(f"Unknown value '{val}' for '{col}'. Defaulting to 0.")
        else:
            row[col] = 0

    row["Monthly Grocery Bill"] = float(inputs["monthly_grocery"])
    row["Vehicle Monthly Distance Km"] = float(inputs["vehicle_km"])
    row["How Long TV PC Daily Hour"] = float(inputs["tv_pc_hours"])
    row["How Many New Clothes Monthly"] = float(inputs["new_clothes"])
    row["How Long Internet Daily Hour"] = float(inputs["internet_hours"])
    row["Waste Bag Weekly Count"] = float(
        str(inputs["waste_bags"]).replace("4 or more", "4")
    )
    row["How Often Did You Recycle"] = float(len(inputs["recycling"]))
    row["Cooking_With"] = float(len(inputs["cooking"]))
    row["vehicle_weekly_km"] = float(inputs["vehicle_km"]) / 4.33
    row["electricity_emission_proxy"] = (
        (float(inputs["monthly_grocery"]) / 5) * 0.82 / 4.33
    )

    df_input = pd.DataFrame([row])
    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = 0.0
    df_input = df_input[feature_names].astype(float)
    return df_input


# ── Predict helper ────────────────────────────────────────────────────────
def predict_emission(inputs):
    df_input = preprocess_input(inputs)
    raw = model.predict(df_input)[0]
    return float(np.expm1(raw) if log_transformed else raw), df_input


# ── Suggestion engine ─────────────────────────────────────────────────────
def get_suggestions(shap_vals, feat_names, inputs):
    vals = (
        shap_vals[0]
        if hasattr(shap_vals, "__len__") and np.array(shap_vals).ndim > 1
        else np.array(shap_vals).flatten()
    )
    shap_dict = dict(zip(feat_names, vals))
    suggestions = []

    if shap_dict.get("Transport", 0) > 0 and inputs["transport"] == "private":
        km = inputs["vehicle_km"]
        saving = round(km * 0.21 * 0.7 / 4.33, 1)
        suggestions.append(
            {
                "icon": "🚌",
                "title": "Switch to public transport",
                "detail": f"You drive ~{km} km/month privately. Switching to public transport 3+ days/week could save around **{saving} kg CO₂/week**.",
                "priority": abs(shap_dict["Transport"]),
                "tips": [
                    "Use metro, bus, or shared auto for your daily commute",
                    "Plan trips in batches to avoid multiple short rides",
                    "Try cycling or walking for distances under 2 km",
                    "Use carpooling apps when driving is unavoidable",
                ],
            }
        )

    if shap_dict.get("Vehicle Type", 0) > 0 and inputs["vehicle_type"] in [
        "petrol",
        "diesel",
    ]:
        saving = round(inputs["vehicle_km"] * 0.09 / 4.33, 1)
        suggestions.append(
            {
                "icon": "⚡",
                "title": "Consider an EV or hybrid",
                "detail": f"Your {inputs['vehicle_type']} vehicle emits ~21% more than a hybrid. At {inputs['vehicle_km']} km/month, switching could save ~**{saving} kg CO₂/week**.",
                "priority": abs(shap_dict["Vehicle Type"]),
                "tips": [
                    "Compare EV/hybrid running costs — cheaper per km in India",
                    "Check FAME-II government subsidies on electric two-wheelers and cars",
                    "For city rides, an e-scooter or e-bike is a great first step",
                    "Charge during off-peak hours (late night) to use greener grid electricity",
                ],
            }
        )

    if shap_dict.get("Frequency of Traveling by Air", 0) > 0 and inputs[
        "air_travel"
    ] in ["frequently", "very frequently"]:
        suggestions.append(
            {
                "icon": "✈️",
                "title": "Reduce air travel",
                "detail": f"You fly **{inputs['air_travel']}** — each return flight emits ~255 kg CO₂. Consider train travel or video calls for shorter trips.",
                "priority": abs(shap_dict.get("Frequency of Traveling by Air", 0)),
                "tips": [
                    "Replace short-haul flights (under 500 km) with train journeys",
                    "Use video calls for business meetings instead of flying",
                    "Choose direct routes — takeoff & landing burn the most fuel",
                    "Travel light to reduce aircraft fuel load",
                ],
            }
        )

    if shap_dict.get("Diet", 0) > 0 and inputs["diet"] in ["omnivore", "pescatarian"]:
        if inputs["diet"] == "omnivore":
            saving = round(200 / 52, 1)
            suggestion_text = f"You eat an omnivore diet. Cutting meat by 3 meals/week saves ~**{saving} kg CO₂/week** (~200 kg/year)."
        else:
            saving = round(100 / 52, 1)
            suggestion_text = f"Switching from pescatarian to vegetarian could save ~**{saving} kg CO₂/week**."
        suggestions.append(
            {
                "icon": "🥗",
                "title": "Reduce meat consumption",
                "detail": suggestion_text,
                "priority": abs(shap_dict["Diet"]),
                "tips": [
                    "Try 'Meatless Monday' — skip meat just one day a week to start",
                    "Replace meat meals with dal, rajma, paneer, or tofu",
                    "Choose locally grown vegetables over packaged or imported produce",
                    "Reduce red meat first — beef and mutton have the highest carbon footprint",
                ],
            }
        )

    if shap_dict.get("Heating Energy Source", 0) > 0 and inputs["heating_source"] in [
        "coal",
        "wood",
    ]:
        suggestions.append(
            {
                "icon": "🔥",
                "title": "Switch heating source",
                "detail": f"You use **{inputs['heating_source']}** for heating — one of the dirtiest sources. Switching to electricity or gas can cut home emissions by up to 40%.",
                "priority": abs(shap_dict.get("Heating Energy Source", 0)),
                "tips": [
                    "Switch to an electric induction stove or microwave for cooking",
                    "Avoid burning wood or coal indoors — it also harms air quality",
                    "Use solar water heaters if available in your area",
                    "Insulate windows and doors to retain heat and reduce energy use",
                ],
            }
        )

    if shap_dict.get("Energy efficiency", 0) > 0 and inputs["energy_eff"] == "No":
        suggestions.append(
            {
                "icon": "🏠",
                "title": "Improve home energy efficiency",
                "detail": "You haven't adopted energy efficiency measures. LED bulbs, insulation, and a smart thermostat can reduce home energy use by **20–30%**.",
                "priority": abs(shap_dict.get("Energy efficiency", 0)),
                "tips": [
                    "Replace bulbs with 5-star BEE-rated LED lights (saves ~80% on lighting)",
                    "Unplug chargers and appliances on standby — they drain power silently",
                    "Set AC to 24°C — each degree lower adds ~6% to electricity consumption",
                    "Use a ceiling fan before switching on the AC",
                ],
            }
        )

    waste_num = int(str(inputs["waste_bags"]).replace("4 or more", "4"))
    if shap_dict.get("Waste Bag Weekly Count", 0) > 0 and waste_num >= 3:
        saving = round(waste_num * 0.5 * 0.5, 1)
        suggestions.append(
            {
                "icon": "♻️",
                "title": "Reduce waste",
                "detail": f"You generate **{inputs['waste_bags']} bags/week** of {inputs['waste_type']} waste. Composting and recycling more could halve this, saving ~**{saving} kg CO₂/week**.",
                "priority": abs(shap_dict.get("Waste Bag Weekly Count", 0)),
                "tips": [
                    "Separate dry waste (plastic, paper, metal) from wet waste (food scraps) daily",
                    "Rinse plastic containers before disposing — contaminated plastic can't be recycled",
                    "Give paper, cardboard, and metal to your local kabadiwala for recycling",
                    "Drop e-waste (phones, batteries, chargers) at authorized collection centres",
                ],
            }
        )

    if len(inputs["recycling"]) == 0:
        suggestions.append(
            {
                "icon": "🗂️",
                "title": "Start recycling",
                "detail": "You currently don't recycle anything. Starting with paper and plastic alone can meaningfully reduce landfill emissions.",
                "priority": 0.5,
                "tips": [
                    "Start small — just separate paper and plastic from the rest of your trash",
                    "Use two bins at home: one for dry recyclables, one for wet/food waste",
                    "Many Indian cities have door-to-door dry waste pickup — check your local schedule",
                    "Avoid single-use plastics: carry a cloth bag, reusable bottle, and steel straw",
                ],
            }
        )

    if (
        shap_dict.get("How Many New Clothes Monthly", 0) > 0
        and inputs["new_clothes"] >= 5
    ):
        suggestions.append(
            {
                "icon": "👕",
                "title": "Buy fewer new clothes",
                "detail": f"You buy **{inputs['new_clothes']} new items/month**. Fast fashion is carbon-heavy. Buying secondhand or halving purchases saves ~15 kg CO₂/month.",
                "priority": abs(shap_dict.get("How Many New Clothes Monthly", 0)),
                "tips": [
                    "Shop second-hand on OLX, Facebook Marketplace, Spoyl, or Swop",
                    "Look for clothes made from recycled or organic (GOTS-certified) fabrics",
                    "Rent outfits for weddings and occasions instead of buying new",
                    "Repair and tailor worn clothes — try a 'no-buy month' once a quarter",
                ],
            }
        )

    if shap_dict.get("How Long TV PC Daily Hour", 0) > 0 and inputs["tv_pc_hours"] >= 6:
        suggestions.append(
            {
                "icon": "💡",
                "title": "Reduce screen time",
                "detail": f"You use TV/PC for **{inputs['tv_pc_hours']} hrs/day**. Cutting by 2 hours saves ~15 kg CO₂/year in electricity.",
                "priority": abs(shap_dict.get("How Long TV PC Daily Hour", 0)),
                "tips": [
                    "Set a daily screen time limit in your phone settings",
                    "Turn off monitors and TVs fully instead of leaving on standby",
                    "Stream at SD/HD instead of 4K — it uses 7× less data and energy",
                    "Replace 1 hour of TV/PC with an outdoor or offline activity each day",
                ],
            }
        )

    suggestions = sorted(suggestions, key=lambda x: x["priority"], reverse=True)

    generic_tips = {
        "Transport": {
            "icon": "🚌",
            "title": "Rethink your transport",
            "detail": f"Transport is one of your top emission drivers. Even reducing private vehicle use 2 days/week makes a meaningful difference.",
            "tips": [
                "Use metro, bus, or shared auto for your daily commute",
                "Plan trips in batches to avoid multiple short rides",
                "Try cycling or walking for distances under 2 km",
                "Use carpooling apps when driving is unavoidable",
            ],
        },
        "Vehicle Monthly Distance Km": {
            "icon": "🚗",
            "title": "Drive less",
            "detail": f"You drive **{inputs['vehicle_km']} km/month**. Combining trips or working from home 1 day/week can cut this noticeably.",
            "tips": [
                "Combine errands into one trip instead of multiple short drives",
                "Work from home at least 1 day a week if your job allows",
                "Use Google Maps to find the most fuel-efficient route",
                "Keep tyre pressure correct — underinflated tyres increase fuel use by ~3%",
            ],
        },
        "vehicle_weekly_km": {
            "icon": "🚗",
            "title": "Reduce weekly driving",
            "detail": "Your weekly driving distance is a significant emission source. Try public transport or cycling for short trips.",
            "tips": [
                "Combine errands into one trip instead of multiple short drives",
                "Work from home at least 1 day a week if your job allows",
                "Use Google Maps to find the most fuel-efficient route",
                "Keep tyre pressure correct — underinflated tyres increase fuel use by ~3%",
            ],
        },
        "Diet": {
            "icon": "🥗",
            "title": "Adjust your diet",
            "detail": f"Your **{inputs['diet']}** diet contributes to your footprint. Even one plant-based day per week reduces food emissions by ~15%.",
            "tips": [
                "Try 'Meatless Monday' — skip meat just one day a week to start",
                "Replace meat meals with dal, rajma, paneer, or tofu",
                "Choose locally grown vegetables over packaged or imported produce",
                "Reduce red meat first — it has the highest carbon footprint",
            ],
        },
        "Frequency of Traveling by Air": {
            "icon": "✈️",
            "title": "Fly less",
            "detail": "Air travel is one of the highest per-trip emission sources. Each return flight adds ~255 kg CO₂.",
            "tips": [
                "Replace short-haul flights (under 500 km) with train journeys",
                "Use video calls for business meetings instead of flying",
                "Choose direct routes — takeoff & landing burn the most fuel",
                "Travel light to reduce aircraft fuel load",
            ],
        },
        "Heating Energy Source": {
            "icon": "🔥",
            "title": "Greener heating",
            "detail": f"**{inputs['heating_source'].title()}** heating has a high carbon intensity. Switching to electricity or gas reduces home emissions significantly.",
            "tips": [
                "Switch to an electric induction stove or microwave for cooking",
                "Avoid burning wood or coal indoors — it also harms air quality",
                "Use solar water heaters if available in your area",
                "Insulate windows and doors to retain heat and reduce energy use",
            ],
        },
        "How Many New Clothes Monthly": {
            "icon": "👕",
            "title": "Slow down on shopping",
            "detail": f"You buy **{inputs['new_clothes']} new clothes/month**. Fast fashion is carbon-heavy — try secondhand or reducing by a few items.",
            "tips": [
                "Shop second-hand on OLX, Facebook Marketplace, Spoyl, or Swop",
                "Look for clothes made from recycled or organic (GOTS-certified) fabrics",
                "Rent outfits for weddings and occasions instead of buying new",
                "Repair and tailor worn clothes — try a 'no-buy month' once a quarter",
            ],
        },
        "How Long TV PC Daily Hour": {
            "icon": "💡",
            "title": "Reduce screen time",
            "detail": f"**{inputs['tv_pc_hours']} hrs/day** of TV/PC use adds up in electricity. Cutting by 2 hrs/day saves ~15 kg CO₂/year.",
            "tips": [
                "Set a daily screen time limit in your phone settings",
                "Turn off monitors and TVs fully instead of leaving on standby",
                "Stream at SD/HD instead of 4K — it uses 7× less data and energy",
                "Replace 1 hour of TV/PC with an outdoor or offline activity each day",
            ],
        },
        "Waste Bag Weekly Count": {
            "icon": "♻️",
            "title": "Reduce your waste",
            "detail": f"You produce **{inputs['waste_bags']} waste bags/week**. Composting food scraps and recycling more can cut this significantly.",
            "tips": [
                "Separate dry waste (plastic, paper, metal) from wet waste (food scraps) daily",
                "Rinse plastic containers before disposing — contaminated plastic can't be recycled",
                "Give paper, cardboard, and metal to your local kabadiwala for recycling",
                "Drop e-waste (phones, batteries, chargers) at authorized collection centres",
            ],
        },
        "Energy efficiency": {
            "icon": "🏠",
            "title": "Improve home efficiency",
            "detail": f"Your home energy efficiency is **{inputs['energy_eff']}**. LED bulbs, insulation, and a smart thermostat can reduce bills and emissions by 20%.",
            "tips": [
                "Replace bulbs with 5-star BEE-rated LED lights (saves ~80% on lighting)",
                "Unplug chargers and appliances on standby — they drain power silently",
                "Set AC to 24°C — each degree lower adds ~6% to electricity consumption",
                "Use a ceiling fan before switching on the AC",
            ],
        },
        "How Long Internet Daily Hour": {
            "icon": "🌐",
            "title": "Cut digital emissions",
            "detail": f"**{inputs['internet_hours']} hrs/day** online has a carbon cost from data centres. Streaming in lower quality and reducing cloud storage helps.",
            "tips": [
                "Stream music and videos at lower quality when fidelity doesn't matter",
                "Clear unused cloud storage and email attachments regularly",
                "Download content for offline use instead of streaming repeatedly",
                "Unsubscribe from newsletters and delete old emails to reduce server load",
            ],
        },
        "Social Activity": {
            "icon": "🌍",
            "title": "Mindful socialising",
            "detail": f"Going out **{inputs['social_activity']}** contributes indirectly through transport and consumption. Local activities reduce this impact.",
            "tips": [
                "Choose local venues to reduce transport emissions",
                "Walk, cycle, or take public transport to social outings instead of driving",
                "Host gatherings at home occasionally instead of going to restaurants",
                "Avoid single-use plastic cups, plates, and straws at events",
            ],
        },
        "Vehicle Type": {
            "icon": "⚡",
            "title": "Consider a greener vehicle",
            "detail": f"A **{inputs['vehicle_type']}** vehicle has higher emissions than hybrid or electric alternatives. Worth considering at your next upgrade.",
            "tips": [
                "Compare EV/hybrid running costs — cheaper per km in India",
                "Check FAME-II government subsidies on electric two-wheelers and cars",
                "For city rides, an e-scooter or e-bike is a great first step",
                "Charge during off-peak hours (late night) to use greener grid electricity",
            ],
        },
    }

    already_covered = {s["title"] for s in suggestions}
    top_features_by_shap = sorted(
        shap_dict.items(), key=lambda x: abs(x[1]), reverse=True
    )

    for feat, shap_val in top_features_by_shap:
        if len(suggestions) >= 4:
            break
        if feat in generic_tips:
            tip = generic_tips[feat].copy()
            if tip["title"] not in already_covered:
                tip["priority"] = abs(shap_val)
                suggestions.append(tip)
                already_covered.add(tip["title"])

    if not suggestions:
        suggestions = [
            {
                "icon": "🌱",
                "title": "You're doing well!",
                "detail": "Your footprint is relatively low. Keep recycling, eating less meat, and using public transport.",
                "priority": 0,
            }
        ]

    return suggestions[:4]


# ── HERO HEADER ───────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-header">
        <div class="hero-eyebrow">🌿 Environmental Intelligence</div>
        <h1 class="hero-title">Carbon Footprint<br>Predictor</h1>
        <p class="hero-sub">Enter your weekly lifestyle habits to estimate your CO₂ emissions, and see personalised eco-tips powered by XGBoost + SHAP.</p>
        <div class="hero-leaf">🌿</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar inputs ────────────────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div style="text-align:center; padding: 0.5rem 0 1.25rem;">
        <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:800;
                    background: linear-gradient(135deg, #16241c 30%, #3f9e2f 100%);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text; margin-bottom:0.25rem;">
            Your Lifestyle Inputs
        </div>
        <div style="width:40px; height:2px; background: linear-gradient(90deg, #0f7a59, #3f9e2f);
                    margin: 0 auto; border-radius:2px;"></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<p style='color:rgba(22,36,28,0.68); font-size:0.82rem; text-align:center; margin-bottom:1rem;'>Fill in each section, then click Calculate below.</p>",
    unsafe_allow_html=True,
)

with st.sidebar.expander("👤 Personal", expanded=False):
    body_type = st.selectbox(
        "Body type", ["underweight", "normal", "overweight", "obese"]
    )
    sex = st.selectbox("Sex", ["male", "female"])

with st.sidebar.expander("🍽️ Diet & Food", expanded=False):
    diet = st.selectbox("Diet type", ["vegan", "vegetarian", "pescatarian", "omnivore"])
    monthly_grocery = st.slider("Monthly grocery spend (₹)", 500, 20000, 4000, step=500)

with st.sidebar.expander("🚗 Transport", expanded=False):
    transport = st.selectbox("Primary transport", ["public", "walk/bicycle", "private"])
    vehicle_type = st.selectbox(
        "Vehicle type", ["none", "petrol", "diesel", "hybrid", "lpg", "electric"]
    )
    vehicle_km = st.slider("Monthly vehicle distance (km)", 0, 5000, 500, step=50)
    air_travel = st.selectbox(
        "Air travel frequency", ["never", "rarely", "frequently", "very frequently"]
    )

with st.sidebar.expander("🏠 Home & Energy", expanded=False):
    heating_source = st.selectbox(
        "Heating energy source", ["coal", "natural gas", "wood", "electricity"]
    )
    energy_eff = st.selectbox("Home energy efficiency", ["No", "Sometimes", "Yes"])
    shower_freq = st.selectbox(
        "Shower frequency",
        ["less frequently", "daily", "twice a day", "more frequently"],
    )

with st.sidebar.expander("🛍️ Lifestyle & Waste", expanded=False):
    social_activity = st.selectbox(
        "How often do you go out?", ["never", "sometimes", "often"]
    )
    new_clothes = st.slider("New clothes per month", 0, 30, 3)
    waste_bags = st.selectbox("Weekly waste bags", ["0", "1", "2", "3", "4 or more"])
    waste_type = st.selectbox(
        "Waste bag size", ["small", "medium", "large", "extra large"]
    )
    recycling = st.multiselect(
        "What do you recycle?",
        ["Paper", "Plastic", "Glass", "Metal"],
        default=["Plastic"],
    )
    cooking = st.multiselect(
        "Cooking appliances",
        ["Stove", "Oven", "Microwave", "Grill", "Airfryer"],
        default=["Stove", "Microwave"],
    )

with st.sidebar.expander("📺 Screen & Digital", expanded=False):
    tv_pc_hours = st.slider("Daily TV/PC hours", 0, 24, 4)
    internet_hours = st.slider("Daily internet hours", 0, 24, 5)

st.sidebar.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

current_inputs = {
    "body_type": body_type,
    "sex": sex,
    "diet": diet,
    "shower_freq": shower_freq,
    "heating_source": heating_source,
    "transport": transport,
    "vehicle_type": vehicle_type,
    "social_activity": social_activity,
    "monthly_grocery": monthly_grocery,
    "air_travel": air_travel,
    "vehicle_km": vehicle_km,
    "waste_bags": waste_bags,
    "waste_type": waste_type,
    "tv_pc_hours": tv_pc_hours,
    "new_clothes": new_clothes,
    "internet_hours": internet_hours,
    "energy_eff": energy_eff,
    "recycling": recycling,
    "cooking": cooking,
}

# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    [
        "📊  My Footprint",
        "🔍  What's Driving It",
        "🔄  What-If Simulator",
    ]
)

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — My Footprint
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        "<div style='height:0.25rem'></div>",
        unsafe_allow_html=True,
    )

    predict_btn = st.button(
        "🌍  Calculate My Carbon Footprint", type="primary", use_container_width=True
    )

    if predict_btn:
        with st.spinner("Analysing your lifestyle habits…"):
            try:
                emission, df_input = predict_emission(current_inputs)
                emission_annual = emission * 52
                peer_pct = get_peer_percentile(emission, all_train_preds)

                st.session_state["emission"] = emission
                st.session_state["emission_annual"] = emission_annual
                st.session_state["df_input"] = df_input
                st.session_state["inputs"] = current_inputs.copy()
                st.session_state["peer_pct"] = peer_pct
                st.session_state["calculated"] = True
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

    if st.session_state.get("calculated"):
        emission = st.session_state["emission"]
        emission_annual = st.session_state["emission_annual"]
        peer_pct = st.session_state["peer_pct"]

        india_avg = 2190
        global_avg = 4890

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # ── Metric cards ──────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Weekly CO₂", f"{emission:.1f} kg")
        m2.metric("Annual CO₂", f"{emission_annual/1000:.2f} tonnes")
        m3.metric(
            "vs India avg",
            f"{(emission_annual/india_avg - 1)*100:+.0f}%",
            delta_color="inverse",
        )
        m4.metric(
            "vs Global avg",
            f"{(emission_annual/global_avg - 1)*100:+.0f}%",
            delta_color="inverse",
        )

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

        # Peer percentile banner
        if peer_pct is not None:
            if peer_pct >= 50:
                st.markdown(
                    f"""<div class="peer-banner-good">
                        <span style="font-size:1.5rem">🏅</span>
                        <span>You emit less CO₂ than <strong style="color:#3f9e2f">{peer_pct}%</strong> of users in our dataset — <em>better than most!</em></span>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div class="peer-banner-warn">
                        <span style="font-size:1.5rem">📊</span>
                        <span>You emit more CO₂ than <strong style="color:#b8852a">{100 - peer_pct}%</strong> of users in our dataset. See the tips below to improve.</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.divider()

        # ── Gauge chart ───────────────────────────────────────────────
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=round(emission_annual / 1000, 2),
                number={
                    "suffix": " t/yr",
                    "font": {"color": "#16241c", "size": 40, "family": "Syne"},
                },
                delta={
                    "reference": 2.19,
                    "suffix": " t (India avg)",
                    "increasing": {"color": "#c2461f"},
                    "decreasing": {"color": "#0f7a59"},
                    "font": {"family": "DM Sans"},
                },
                gauge={
                    "axis": {
                        "range": [0, 20],
                        "ticksuffix": " t",
                        "tickcolor": "rgba(22,36,28,0.65)",
                        "tickfont": {
                            "color": "rgba(22,36,28,0.65)",
                            "family": "DM Sans",
                        },
                    },
                    "bar": {"color": "#138a64", "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 2], "color": "rgba(15,122,89,0.32)"},
                        {"range": [2, 5], "color": "rgba(184,133,42,0.20)"},
                        {"range": [5, 20], "color": "rgba(194,70,31,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "rgba(22,36,28,0.75)", "width": 2},
                        "thickness": 0.85,
                        "value": 4.89,
                    },
                },
                title={
                    "text": (
                        "<span style='font-family:Syne;font-size:15px;color:#16241c'>Annual CO₂ Emissions</span><br>"
                        "<span style='font-size:11px;color:#0f7a59'>● Below India avg (2.19t)</span>  "
                        "<span style='font-size:11px;color:#b8852a'>● India→Global (2.19–4.89t)</span>  "
                        "<span style='font-size:11px;color:#c2461f'>● Above global avg (4.89t)</span>"
                    ),
                    "font": {"color": "#16241c", "size": 14},
                },
            )
        )
        fig_gauge.update_layout(
            height=380,
            margin=dict(t=110, b=20, l=40, r=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#16241c", family="DM Sans"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        ca, cb = st.columns(2)
        ca.info("🇮🇳 India average: **1.9 tonnes/year** (36.5 kg/week)")
        cb.info("🌍 Global average: **4.5 tonnes/year** (86.5 kg/week)")

        # ── Eco tips ──────────────────────────────────────────────────
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-label'>🌱 Personalised Eco-Tips</div>",
            unsafe_allow_html=True,
        )

        df_input = st.session_state["df_input"]
        shap_vals = explainer.shap_values(df_input)
        suggestions = get_suggestions(shap_vals, feature_names, current_inputs)

        tip_cols = st.columns(2)
        for i, tip in enumerate(suggestions):
            with tip_cols[i % 2]:
                detail_html = re.sub(
                    r"\*\*(.*?)\*\*", r"<strong>\1</strong>", tip["detail"]
                )
                tips_html = "".join(f"<li>{t}</li>" for t in tip.get("tips", []))
                card_html = (
                    f'<div class="eco-tip-card">'
                    f'<div class="eco-tip-icon">{tip["icon"]}</div>'
                    f'<div class="eco-tip-title">{tip["title"]}</div>'
                    f'<div class="eco-tip-detail">{detail_html}</div>'
                    f'<div class="eco-tip-divider"></div>'
                    f'<div class="eco-tip-tips-label">Action Steps</div>'
                    f'<ul class="eco-tip-list">{tips_html}</ul>'
                    f"</div>"
                )
                st.markdown(card_html, unsafe_allow_html=True)

        # ── Download report ───────────────────────────────────────────
        st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
        st.divider()

        report_lines = [
            "Carbon Footprint Report",
            "=" * 40,
            f"Weekly emission   : {emission:.1f} kg CO2",
            f"Annual emission   : {emission_annual/1000:.2f} tonnes CO2",
            f"vs India avg      : {(emission_annual/india_avg - 1)*100:+.0f}%",
            f"vs Global avg     : {(emission_annual/global_avg - 1)*100:+.0f}%",
        ]
        if peer_pct is not None:
            report_lines.append(f"Peer comparison   : Better than {peer_pct}% of users")
        report_lines += ["", "Top eco-tips:", "-" * 30]
        for tip in suggestions:
            clean_detail = tip["detail"].replace("**", "")
            report_lines.append(f"{tip['icon']} {tip['title']}: {clean_detail}")
            for step in tip.get("tips", []):
                clean_step = re.sub(r"<.*?>", "", step)
                report_lines.append(f"    - {clean_step}")
            report_lines.append("")
        report_text = "\n".join(report_lines)

        st.download_button(
            "📄  Download My Carbon Report",
            report_text,
            file_name="my_carbon_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown(
            """<div style="
                background: linear-gradient(135deg, rgba(255,255,255,0.90) 0%, rgba(15,122,89,0.08) 100%);
                border: 1px dashed rgba(15,122,89,0.32);
                border-radius: 20px;
                padding: 3rem 2rem;
                text-align: center;
            ">
                <div style="font-size:3rem; margin-bottom:1rem; opacity:0.6">🌿</div>
                <div style="font-family:'Syne',sans-serif; font-size:1.15rem; font-weight:700;
                            color:rgba(22,36,28,0.82); margin-bottom:0.5rem;">
                    Ready to measure your impact?
                </div>
                <div style="font-family:'DM Sans',sans-serif; font-size:0.9rem;
                            color:rgba(22,36,28,0.65); max-width:380px; margin:0 auto;">
                    Fill in your lifestyle habits in the sidebar, then click <strong style="color:#138a64">Calculate</strong> to see your personalised carbon footprint.
                </div>
            </div>""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — SHAP
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-label'>🔍 Feature Impact Analysis</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.get("calculated"):
        st.markdown(
            """<div style="
                background: rgba(15,122,89,0.08);
                border: 1px dashed rgba(15,122,89,0.25);
                border-radius: 14px;
                padding: 2rem;
                text-align: center;
                color: rgba(22,36,28,0.70);
                font-family: 'DM Sans', sans-serif;
            ">
                Calculate your footprint in the <strong style="color:#138a64">My Footprint</strong> tab to unlock this analysis.
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        df_input = st.session_state["df_input"]
        shap_vals = explainer.shap_values(df_input)

        vals = (
            shap_vals[0]
            if hasattr(shap_vals, "__len__") and np.array(shap_vals).ndim > 1
            else np.array(shap_vals).flatten()
        )
        shap_series = (
            pd.Series(vals, index=feature_names)
            .sort_values(key=abs, ascending=False)
            .head(10)
        )

        colors = ["#c2461f" if v > 0 else "#0f7a59" for v in shap_series.values]
        fig_shap = go.Figure(
            go.Bar(
                x=shap_series.values,
                y=shap_series.index,
                orientation="h",
                marker_color=colors,
                marker_line_width=0,
                text=[f"{v:+.3f}" for v in shap_series.values],
                textposition="outside",
                textfont=dict(color="rgba(22,36,28,0.75)", family="DM Sans", size=11),
            )
        )
        fig_shap.update_layout(
            title={
                "text": "Top 10 Features — Impact on Your Predicted Emission",
                "font": {"color": "#16241c", "size": 14, "family": "Syne"},
            },
            xaxis_title="SHAP value (contribution to CO₂ prediction)",
            xaxis=dict(
                color="rgba(22,36,28,0.68)",
                gridcolor="rgba(15,122,89,0.14)",
                zerolinecolor="rgba(15,122,89,0.38)",
                tickfont=dict(family="DM Sans"),
            ),
            yaxis=dict(
                color="rgba(22,36,28,0.85)",
                autorange="reversed",
                tickfont=dict(family="DM Sans"),
            ),
            height=460,
            margin=dict(t=60, b=30, l=20, r=90),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.75)",
            font=dict(color="rgba(22,36,28,0.82)", family="DM Sans"),
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.markdown(
            "<p style='font-family:DM Sans; font-size:0.8rem; color:rgba(22,36,28,0.65); margin-top:-0.5rem;'>"
            "🔴 Red = increases your emission &nbsp;·&nbsp; 🟢 Green = reduces it &nbsp;·&nbsp; Bar length = magnitude of impact</p>",
            unsafe_allow_html=True,
        )

        with st.expander("📋  Full feature contribution table"):
            full_vals = (
                shap_vals[0]
                if np.array(shap_vals).ndim > 1
                else np.array(shap_vals).flatten()
            )
            full_df = (
                pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "SHAP value": [round(v, 4) for v in full_vals],
                        "Impact": [
                            "↑ Increases" if v > 0 else "↓ Decreases" for v in full_vals
                        ],
                    }
                )
                .sort_values("SHAP value", key=abs, ascending=False)
                .reset_index(drop=True)
            )

            rows_html = ""
            for i, row in full_df.iterrows():
                impact_color = "#d6492a" if row["Impact"].startswith("↑") else "#138a64"
                row_bg = "#fbfdf9" if i % 2 == 0 else "#eef6f0"
                rows_html += f"""
                <tr class="shap-row">
                    <td class="shap-feat" style="background:{row_bg} !important;">{row['Feature']}</td>
                    <td class="shap-val" style="color:{impact_color} !important; background:{row_bg} !important;">{row['SHAP value']:+.4f}</td>
                    <td class="shap-imp" style="color:{impact_color} !important; background:{row_bg} !important;">{row['Impact']}</td>
                </tr>"""

            st.markdown(
                f"""
                <style>
                    .shap-table-wrap {{
                        max-height: 420px;
                        overflow-y: auto;
                        border-radius: 12px;
                        border: 1px solid rgba(15,122,89,0.25);
                        background: #fbfdf9;
                    }}
                    .shap-table {{
                        width: 100%;
                        border-collapse: collapse;
                    }}
                    .shap-table thead tr {{
                        background: rgba(15,122,89,0.16) !important;
                    }}
                    .shap-table thead th {{
                        padding: 0.6rem 0.75rem;
                        font-family: 'Syne', sans-serif !important;
                        font-size: 0.75rem !important;
                        font-weight: 700 !important;
                        letter-spacing: 0.08em;
                        text-transform: uppercase;
                        color: #138a64 !important;
                        border-bottom: 1px solid rgba(15,122,89,0.32);
                    }}
                    .shap-feat {{
                        padding: 0.5rem 0.75rem !important;
                        font-family: 'DM Sans', sans-serif !important;
                        font-size: 0.85rem !important;
                        color: #16241c !important;
                        border-bottom: 1px solid rgba(15,122,89,0.12) !important;
                    }}
                    .shap-val {{
                        padding: 0.5rem 0.75rem !important;
                        text-align: right !important;
                        font-family: 'DM Sans', sans-serif !important;
                        font-size: 0.85rem !important;
                        font-weight: 700 !important;
                        border-bottom: 1px solid rgba(15,122,89,0.12) !important;
                    }}
                    .shap-imp {{
                        padding: 0.5rem 0.75rem !important;
                        text-align: center !important;
                        font-family: 'DM Sans', sans-serif !important;
                        font-size: 0.82rem !important;
                        border-bottom: 1px solid rgba(15,122,89,0.12) !important;
                    }}
                </style>
                <div class="shap-table-wrap">
                    <table class="shap-table">
                        <thead>
                            <tr>
                                <th style="text-align:left">Feature</th>
                                <th style="text-align:right">SHAP Value</th>
                                <th style="text-align:center">Impact</th>
                            </tr>
                        </thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — What-If Simulator
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-label'>🔄 What-If Simulator</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-family:DM Sans; font-size:0.9rem; color:rgba(22,36,28,0.70); margin-bottom:1rem;'>Adjust habits below to see how your carbon footprint would change in real-time.</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.get("calculated"):
        st.markdown(
            """<div style="
                background: rgba(15,122,89,0.08);
                border: 1px dashed rgba(15,122,89,0.25);
                border-radius: 14px;
                padding: 2rem;
                text-align: center;
                color: rgba(22,36,28,0.70);
                font-family: 'DM Sans', sans-serif;
            ">
                Calculate your footprint in the <strong style="color:#138a64">My Footprint</strong> tab first.
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        base_emission = st.session_state["emission"]
        base_inputs = st.session_state["inputs"].copy()

        st.markdown(
            f"""<div class="baseline-box">
                📍 Baseline — Weekly: <span>{base_emission:.1f} kg CO₂</span>
            </div>""",
            unsafe_allow_html=True,
        )

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown(
                "<div class='wi-category'>🚗 Transport</div>", unsafe_allow_html=True
            )
            wi_transport = st.selectbox(
                "Transport mode",
                ["public", "walk/bicycle", "private"],
                index=["public", "walk/bicycle", "private"].index(
                    base_inputs["transport"]
                ),
                key="wi_transport",
            )
            wi_vehicle_km = st.slider(
                "Monthly vehicle km",
                0,
                5000,
                int(base_inputs["vehicle_km"]),
                step=50,
                key="wi_km",
            )
            wi_vehicle_type = st.selectbox(
                "Vehicle type",
                ["none", "petrol", "diesel", "hybrid", "lpg", "electric"],
                index=["none", "petrol", "diesel", "hybrid", "lpg", "electric"].index(
                    base_inputs["vehicle_type"]
                ),
                key="wi_vtype",
            )
            wi_air = st.selectbox(
                "Air travel frequency",
                ["never", "rarely", "frequently", "very frequently"],
                index=["never", "rarely", "frequently", "very frequently"].index(
                    base_inputs["air_travel"]
                ),
                key="wi_air",
            )

        with col_b:
            st.markdown(
                "<div class='wi-category'>🍽️ Diet & Home</div>", unsafe_allow_html=True
            )
            wi_diet = st.selectbox(
                "Diet",
                ["vegan", "vegetarian", "pescatarian", "omnivore"],
                index=["vegan", "vegetarian", "pescatarian", "omnivore"].index(
                    base_inputs["diet"]
                ),
                key="wi_diet",
            )
            wi_heating = st.selectbox(
                "Heating source",
                ["coal", "natural gas", "wood", "electricity"],
                index=["coal", "natural gas", "wood", "electricity"].index(
                    base_inputs["heating_source"]
                ),
                key="wi_heating",
            )
            wi_social = st.selectbox(
                "How often go out?",
                ["never", "sometimes", "often"],
                index=["never", "sometimes", "often"].index(
                    base_inputs["social_activity"]
                ),
                key="wi_social",
            )
            wi_grocery = st.slider(
                "Monthly grocery bill (₹)",
                500,
                20000,
                int(base_inputs["monthly_grocery"]),
                step=500,
                key="wi_grocery",
            )
            wi_recycling = st.multiselect(
                "What do you recycle?",
                ["Paper", "Plastic", "Glass", "Metal"],
                default=list(base_inputs["recycling"]),
                key="wi_recycling",
            )

        with col_c:
            st.markdown(
                "<div class='wi-category'>🛍️ Lifestyle & Waste</div>",
                unsafe_allow_html=True,
            )
            wi_waste_bags = st.selectbox(
                "Weekly waste bags",
                ["0", "1", "2", "3", "4 or more"],
                index=["0", "1", "2", "3", "4 or more"].index(
                    str(base_inputs["waste_bags"])
                ),
                key="wi_waste_bags",
            )
            wi_waste_type = st.selectbox(
                "Waste bag size",
                ["small", "medium", "large", "extra large"],
                index=["small", "medium", "large", "extra large"].index(
                    base_inputs["waste_type"]
                ),
                key="wi_waste_type",
            )
            wi_new_clothes = st.slider(
                "New clothes/month",
                0,
                30,
                int(base_inputs["new_clothes"]),
                key="wi_clothes",
            )
            wi_internet = st.slider(
                "Daily internet hours",
                0,
                24,
                int(base_inputs["internet_hours"]),
                key="wi_internet",
            )

        st.divider()

        new_inputs = base_inputs.copy()
        new_inputs.update(
            {
                "transport": wi_transport,
                "vehicle_km": wi_vehicle_km,
                "vehicle_type": wi_vehicle_type,
                "air_travel": wi_air,
                "diet": wi_diet,
                "heating_source": wi_heating,
                "social_activity": wi_social,
                "monthly_grocery": wi_grocery,
                "recycling": wi_recycling,
                "waste_bags": (
                    int(wi_waste_bags)
                    if str(wi_waste_bags).isdigit()
                    else wi_waste_bags
                ),
                "waste_type": wi_waste_type,
                "new_clothes": wi_new_clothes,
                "internet_hours": wi_internet,
            }
        )

        try:
            new_emission, _ = predict_emission(new_inputs)
            delta_weekly = base_emission - new_emission
            delta_annual = delta_weekly * 52

            r1, r2, r3 = st.columns(3)
            r1.metric(
                "New weekly emission",
                f"{new_emission:.1f} kg",
                delta=f"{-delta_weekly:+.1f} kg vs current",
                delta_color="inverse",
            )
            r2.metric("New annual emission", f"{new_emission*52/1000:.2f} tonnes")
            r3.metric(
                "Annual CO₂ saved",
                f"{delta_annual:.0f} kg",
                delta_color="normal" if delta_annual >= 0 else "inverse",
            )

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

            if delta_weekly > 0:
                pct = (delta_weekly / base_emission) * 100
                st.success(
                    f"✅  These changes reduce your weekly footprint by **{delta_weekly:.1f} kg ({pct:.1f}%)** — saving **{delta_annual:.0f} kg CO₂/year**"
                )
            elif delta_weekly < 0:
                st.warning(
                    f"⚠️  These changes would **increase** your footprint by {abs(delta_weekly):.1f} kg/week."
                )
            else:
                st.info("No change in footprint with these settings.")

            # ── Per-change breakdown table ────────────────────────────
            changes_made = []
            change_map = {
                "transport": ("Transport mode", base_inputs["transport"], wi_transport),
                "vehicle_km": (
                    "Monthly vehicle km",
                    base_inputs["vehicle_km"],
                    wi_vehicle_km,
                ),
                "vehicle_type": (
                    "Vehicle type",
                    base_inputs["vehicle_type"],
                    wi_vehicle_type,
                ),
                "air_travel": ("Air travel", base_inputs["air_travel"], wi_air),
                "diet": ("Diet", base_inputs["diet"], wi_diet),
                "heating_source": (
                    "Heating source",
                    base_inputs["heating_source"],
                    wi_heating,
                ),
                "social_activity": (
                    "Social activity",
                    base_inputs["social_activity"],
                    wi_social,
                ),
                "monthly_grocery": (
                    "Monthly grocery (₹)",
                    base_inputs["monthly_grocery"],
                    wi_grocery,
                ),
                "recycling": (
                    "Recycling",
                    base_inputs["recycling"],
                    wi_recycling,
                ),
                "waste_bags": (
                    "Waste bags/week",
                    base_inputs["waste_bags"],
                    wi_waste_bags,
                ),
                "waste_type": (
                    "Waste bag size",
                    base_inputs["waste_type"],
                    wi_waste_type,
                ),
                "new_clothes": (
                    "New clothes/month",
                    base_inputs["new_clothes"],
                    wi_new_clothes,
                ),
                "internet_hours": (
                    "Internet hours/day",
                    base_inputs["internet_hours"],
                    wi_internet,
                ),
            }

            def _norm(v):
                if isinstance(v, list):
                    return sorted([str(x) for x in v])
                try:
                    return round(float(v), 4)
                except (ValueError, TypeError):
                    return str(v).strip().lower()

            for key, (label, old_val, new_val) in change_map.items():
                if _norm(old_val) != _norm(new_val):
                    changes_made.append(
                        {
                            "Parameter": label,
                            "Current": str(old_val),
                            "What-If": str(new_val),
                        }
                    )

            if changes_made:
                with st.expander("📋  Changes you made", expanded=True):
                    st.table(pd.DataFrame(changes_made).set_index("Parameter"))

            # ── Comparison chart ──────────────────────────────────────
            fig_compare = go.Figure()
            fig_compare.add_trace(
                go.Bar(
                    name="Current",
                    x=["Weekly CO₂ (kg)"],
                    y=[round(base_emission, 1)],
                    marker_color="#5b52c9",
                    marker_line_width=0,
                    text=[f"{base_emission:.1f}"],
                    textposition="outside",
                    textfont=dict(color="rgba(22,36,28,0.82)", family="Syne", size=14),
                )
            )
            fig_compare.add_trace(
                go.Bar(
                    name="With changes",
                    x=["Weekly CO₂ (kg)"],
                    y=[round(new_emission, 1)],
                    marker_color=(
                        "#0f7a59" if new_emission < base_emission else "#c2461f"
                    ),
                    marker_line_width=0,
                    text=[f"{new_emission:.1f}"],
                    textposition="outside",
                    textfont=dict(color="rgba(22,36,28,0.82)", family="Syne", size=14),
                )
            )
            fig_compare.update_layout(
                barmode="group",
                height=340,
                title={
                    "text": "Current vs Changed Habits",
                    "font": {"color": "#16241c", "size": 14, "family": "Syne"},
                },
                margin=dict(t=55, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.75)",
                font=dict(color="rgba(22,36,28,0.75)", family="DM Sans"),
                xaxis=dict(
                    gridcolor="rgba(15,122,89,0.14)",
                    zerolinecolor="rgba(15,122,89,0.25)",
                ),
                yaxis=dict(
                    gridcolor="rgba(15,122,89,0.14)",
                    zerolinecolor="rgba(15,122,89,0.25)",
                ),
                legend=dict(
                    font=dict(color="rgba(22,36,28,0.82)", family="DM Sans"),
                    bgcolor="rgba(255,255,255,0.90)",
                    bordercolor="rgba(15,122,89,0.25)",
                    borderwidth=1,
                ),
            )
            st.plotly_chart(fig_compare, use_container_width=True)

        except Exception as e:
            st.error(f"Simulation error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
st.divider()
st.markdown(
    """<div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.5rem;">
        <span style="font-family:'DM Sans',sans-serif; font-size:0.75rem; color:rgba(22,36,28,0.55);">
            🌍 Carbon Footprint Predictor
        </span>
        <span style="font-family:'DM Sans',sans-serif; font-size:0.75rem; color:rgba(22,36,28,0.55);">
            Model: XGBoost + SHAP &nbsp;·&nbsp; Dataset: Kaggle Individual Carbon Footprint
        </span>
    </div>""",
    unsafe_allow_html=True,
)
