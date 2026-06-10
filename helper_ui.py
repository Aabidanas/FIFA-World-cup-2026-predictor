import streamlit as st

def inject_global_css():
    st.markdown("""
    <style>
    /* --- 3D FLIP CARD ENGINE --- */
    .flip-container {
        perspective: 1000px;
        display: block;
        width: 100%;
        height: 240px; 
        cursor: pointer;
        margin-bottom: 10px;
    }
    .flip-container input[type="checkbox"]:checked ~ .flipper {
        transform: rotateY(180deg);
    }
    .flipper {
        transition: 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        transform-style: preserve-3d;
        position: relative;
        width: 100%;
        height: 100%;
    }
    .front, .back {
        backface-visibility: hidden;
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
    }
    .back {
        transform: rotateY(180deg);
        background: var(--bg-elevated);
        border-color: #333333;
    }
    .back:hover {
        border-color: var(--text-muted);
    }

    /* --- GLOBAL TOKENS & RESET --- */
    :root {
        --bg-base: #0A0A0A; --bg-surface: #111111; --bg-elevated: #1A1A1A; --bg-subtle: #222222;
        --text-primary: #F0F0F0; --text-secondary: #A0A0A0; --text-muted: #555555;
        --accent: #D9541E; --accent-dim: #A64216;
        --radius-sm: 4px; --radius-md: 8px; --radius-lg: 16px;
        --font-display: 'Inter', system-ui, sans-serif; --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
        --shadow-card: 0 1px 3px rgba(0,0,0,0.5), 0 4px 16px rgba(0,0,0,0.3);
        --transition: all 0.18s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    html, body, [data-testid="stAppViewContainer"] { background: var(--bg-base); }
    .main .block-container { max-width: 1200px; padding: 2rem 2.5rem; margin: 0 auto; }
    
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stHeader"] * { color: var(--text-primary) !important; fill: var(--text-primary) !important; }
    footer { visibility: hidden; } 
    
    /* Deep Orange Title Header Accent */
    h1 { font: 700 2rem/1.2 var(--font-display); color: var(--accent); letter-spacing: -0.03em; }
    h2 { font: 600 1.25rem/1.3 var(--font-display); color: var(--text-primary); letter-spacing: -0.02em; }
    h3 { font: 500 1rem/1.4 var(--font-display); color: var(--accent-dim); letter-spacing: 0.05em; text-transform: uppercase; font-size: 0.75rem; }
    p, li { font: 400 0.9375rem/1.65 var(--font-display); color: var(--text-secondary); }
    code, pre { font-family: var(--font-mono); font-size: 0.85em; }
    
    [data-testid="stSidebar"] { background: var(--bg-surface) !important; border-right: 1px solid var(--bg-subtle); padding: 1.5rem 1rem; }
    
    /* Deep Orange Button Mapping */
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: var(--accent) !important; color: #FFFFFF !important; border: none !important; border-radius: var(--radius-md) !important;
        font-weight: 600 !important; font-size: 0.875rem !important; padding: 0.6rem 1.4rem !important; letter-spacing: 0.01em !important; transition: var(--transition) !important; width: 100%;
        box-shadow: 0 0 12px rgba(217, 84, 30, 0.2);
    }
    div[data-testid="stButton"] button[kind="primary"]:hover { background-color: #F06225 !important; transform: translateY(-1px) !important; box-shadow: 0 0 18px rgba(217, 84, 30, 0.4); }
    
    [data-testid="stMetric"] { background: var(--bg-surface); border: 1px solid var(--bg-subtle); border-radius: var(--radius-lg); padding: 1.25rem 1.5rem; box-shadow: var(--shadow-card); }
    [data-testid="stExpander"] { border: 1px solid var(--bg-subtle) !important; border-radius: var(--radius-md) !important; background: var(--bg-surface) !important; }
    [data-testid="stExpander"] summary { color: var(--text-secondary); font-size: 0.875rem; font-weight: 500; font-family: var(--font-mono); letter-spacing: 1px; }
    
    /* Interactive Tabs Accent Color */
    [data-testid="stTabs"] [role="tab"] { color: var(--text-muted); border-bottom: 2px solid transparent; font-size: 0.875rem; padding: 0.5rem 1rem; transition: var(--transition); font-family: var(--font-mono); letter-spacing: 1px; }
    [data-testid="stTabs"] [aria-selected="true"] { color: var(--text-primary); border-bottom-color: var(--accent); }
    
    [data-testid="stProgress"] > div > div { background: linear-gradient(90deg, var(--accent), var(--accent-dim)) !important; border-radius: 99px !important; }
    
    .card { background: var(--bg-surface); border: 1px solid #1E1E1E; border-radius: var(--radius-lg); padding: 1.5rem; box-shadow: var(--shadow-card); transition: var(--transition); height: 100%; }
    .card:hover { border-color: var(--accent-dim); }
    .card-label { color: var(--accent-dim); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem; font-family: var(--font-mono); font-weight: 600; }
    </style>
    
    <script>
    document.querySelectorAll('.card').forEach((el, i) => {
        el.style.opacity = 0; el.style.transform = 'translateY(16px)';
        el.style.transition = `opacity 0.4s ease ${i*0.07}s, transform 0.4s ease ${i*0.07}s`;
        setTimeout(() => { el.style.opacity = 1; el.style.transform = 'translateY(0)'; }, 50);
    });
    </script>
    """, unsafe_allow_html=True)

def status_dot(color, label):
    colors = {"live": "#D9541E", "idle": "#555", "error": "#CC3333"}
    st.markdown(f"""
    <span style="display:inline-flex;align-items:center;gap:6px;font-size:0.8rem;color:#777;font-family:var(--font-mono)">
      <span style="width:6px;height:6px;border-radius:50%;background:{colors[color]};box-shadow:0 0 6px {colors[color]}66"></span>{label}
    </span>""", unsafe_allow_html=True)

def section_divider(label):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:1rem;margin:2rem 0">
      <div style="flex:1;height:1px;background:#1E1E1E"></div>
      <span style="color:#A64216;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;font-family:var(--font-mono);font-weight:600;">{label}</span>
      <div style="flex:1;height:1px;background:#1E1E1E"></div>
    </div>""", unsafe_allow_html=True)