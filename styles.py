# styles.py - Visual theme
import streamlit as st

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #F9F9F7;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding: 1.5rem 2rem;
        max-width: 960px;
    }

    .app-header {
        display: flex;
        align-items: baseline;
        gap: 12px;
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid #E5E5E3;
    }
    .app-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1A1A1A;
        letter-spacing: -0.03em;
    }
    .app-subtitle {
        font-size: 0.85rem;
        color: #999;
        font-weight: 400;
        letter-spacing: 0.01em;
    }

    .card {
        background: #FFFFFF;
        border: 1px solid #E5E5E3;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .card-label {
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #999;
        margin-bottom: 0.75rem;
    }

    .emotion-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1A1A1A;
        letter-spacing: -0.03em;
        line-height: 1;
        margin-bottom: 0.25rem;
    }
    .emotion-sub {
        font-size: 0.8rem;
        color: #999;
    }

    .fused-card {
        background: #1A1A1A;
        border: none;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .fused-label {
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #666;
        margin-bottom: 0.75rem;
    }
    .fused-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #FFFFFF;
        letter-spacing: -0.04em;
        line-height: 1;
    }
    .fused-confirmed {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 500;
        color: #4CAF50;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }

    .conf-bar-bg {
        height: 3px;
        background: #F0F0EE;
        border-radius: 2px;
        margin-top: 0.75rem;
    }
    .conf-bar-fill {
        height: 3px;
        background: #1A1A1A;
        border-radius: 2px;
        transition: width 0.3s ease;
    }

    .status-row {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-top: 0.5rem;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #4CAF50;
    }
    .status-dot.inactive {
        background: #CCC;
    }
    .status-text {
        font-size: 0.75rem;
        color: #999;
    }

    .stButton button {
        background: #F0F0EE !important;
        color: #1A1A1A !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.25rem !important;
        letter-spacing: 0.01em !important;
        transition: background 0.2s !important;
    }
    .stButton button:hover {
        background: #E5E5E3 !important;
    }

    .stress-value {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
    }
    .stress-low { color: #4CAF50; }
    .stress-mid { color: #E0A72E; }
    .stress-high { color: #E05A2E; }
    .comparison-text {
        font-size: 0.95rem;
        color: #1A1A1A;
        line-height: 1.4;
    }
    </style>
    """, unsafe_allow_html=True)