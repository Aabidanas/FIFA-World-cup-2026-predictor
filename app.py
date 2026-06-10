import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
import tempfile
import time
import warnings
warnings.filterwarnings('ignore')

from helper_ui import inject_global_css, status_dot, section_divider
from theme import PLOTLY_THEME

st.set_page_config(page_title="FIFA World Cup Predictor", page_icon="◼", layout="wide", initial_sidebar_state="expanded")
inject_global_css()

# --- ZONE A: HEADER ---
col1, col2 = st.columns([5, 1])
with col1:
    st.markdown("<h1>FIFA World Cup Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='margin-top:-10px; color:var(--text-muted); font-family:var(--font-mono); font-size:0.85rem;'>Prediction Model</p>", unsafe_allow_html=True)

# --- PARAMETERS ---
OPTIMAL_DECAY, OPTIMAL_TROPHY, OPTIMAL_UNIFORMITY, OPTIMAL_ELASTICITY, OPTIMAL_PEDIGREE = 0.15, 0.08, 0.02, 0.15, 0.10
def get_code(t): return t[:3].upper()

# --- DATA INGESTION ---
@st.cache_resource
def load_and_process_historical_data():
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    matches = pd.read_csv(url)
    matches['date'] = pd.to_datetime(matches['date'])
    modern_matches = matches[matches['date'] >= '2006-01-01'].copy()
    modern_matches['outcome'] = np.where(modern_matches['home_score'] > modern_matches['away_score'], 2, np.where(modern_matches['home_score'] == modern_matches['away_score'], 1, 0))
    max_date = modern_matches['date'].max()
    elo_ratings = {}
    home_elos, away_elos = np.zeros(len(modern_matches)), np.zeros(len(modern_matches))
    home_arr, away_arr = modern_matches['home_team'].to_numpy(), modern_matches['away_team'].to_numpy()
    outcome_arr, tournament_arr = modern_matches['outcome'].to_numpy(), modern_matches['tournament'].to_numpy()
    date_arr = modern_matches['date'].to_numpy()
    
    for i in range(len(modern_matches)):
        home, away, tourney = home_arr[i], away_arr[i], tournament_arr[i]
        if home not in elo_ratings: elo_ratings[home] = 1500
        if away not in elo_ratings: elo_ratings[away] = 1500
        home_elos[i], away_elos[i] = elo_ratings[home], elo_ratings[away]
        actual = 1.0 if outcome_arr[i] == 2 else 0.5 if outcome_arr[i] == 1 else 0.0
        expected = 1 / (1 + 10 ** ((elo_ratings[away] - elo_ratings[home]) / 400))
        base_k = 50 if tourney == 'FIFA World Cup' else 30 if any(x in tourney for x in ['Qualifiers', 'Euro', 'Copa']) else 10
        adaptive_k = base_k * np.exp(-OPTIMAL_DECAY * (max_date - pd.Timestamp(date_arr[i])).days / 365.25)
        elo_shift = adaptive_k * (actual - expected)
        elo_ratings[home] += elo_shift
        elo_ratings[away] -= elo_shift
        
    modern_matches['home_elo'], modern_matches['away_elo'] = home_elos, away_elos
    modern_matches['elo_difference'] = modern_matches['home_elo'] - modern_matches['away_elo']
    sa_teams = {'Brazil', 'Argentina', 'Uruguay', 'Colombia', 'Chile', 'Peru', 'Ecuador', 'Paraguay'}
    eu_teams = {'Germany', 'France', 'Spain', 'Italy', 'England', 'Portugal', 'Netherlands', 'Croatia', 'Belgium', 'Switzerland'}
    def assign_cont(team): return 'SA' if team in sa_teams else 'EU' if team in eu_teams else 'Other'
    modern_matches['home_continent'] = modern_matches['home_team'].apply(assign_cont)
    modern_matches['away_continent'] = modern_matches['away_team'].apply(assign_cont)
    le = LabelEncoder()
    modern_matches['home_continent_encoded'] = le.fit_transform(modern_matches['home_continent'])
    modern_matches['away_continent_encoded'] = le.transform(modern_matches['away_continent'])
    features = ['home_elo', 'away_elo', 'elo_difference', 'home_continent_encoded', 'away_continent_encoded']
    X, y = modern_matches[features], modern_matches['outcome'].astype(int)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X, y, sample_weight=np.exp(-0.0006 * (modern_matches['date'].max() - modern_matches['date']).dt.days))
    return model, elo_ratings, le, features, assign_cont

xgb_model, elo_ratings, le, features, assign_cont = load_and_process_historical_data()

@st.cache_data
def process_player_squads():
    try:
        fifa_df = pd.read_csv('ea_fc26_players.csv', low_memory=False)
    except FileNotFoundError:
        st.error("Dataset ea_fc26_players.csv missing from root.")
        st.stop()
    fifa_df['value_proxy'] = np.where(fifa_df['overallRating'] >= 70, ((fifa_df['overallRating'] - 60) ** 3.6) * 110000, 450000)
    def agg_stats(group):
        top_11 = group.nlargest(11, 'overallRating')['overallRating']
        return pd.Series({'rating_mean': top_11.mean(), 'rating_std': top_11.std() if len(top_11) >= 11 else 5.0, 'value_m': group.nlargest(23, 'overallRating')['value_proxy'].sum() / 1000000})
    df = fifa_df.groupby('nationality').apply(agg_stats).reset_index()
    df['nationality'] = df['nationality'].replace({'United States': 'USA', 'Republic of Ireland': 'Ireland', 'Korea Republic': 'South Korea', 'Czech Republic': 'Czechia', 'China PR': 'China', 'Bosnia & Herzegovina': 'Bosnia and Herzegovina', 'IR Iran': 'Iran'})
    return df

nt_df = process_player_squads()
trophy_weights = {'Argentina': 0.05, 'Spain': 0.05, 'France': 0.02, 'Italy': 0.01}

# --- PDF GENERATOR ENGINE ---
def build_executive_pdf(results_tree, final_winner, shock_target):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_fill_color(14, 17, 23)
    pdf.rect(0, 0, 210, 297, 'F')
    
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Courier", 'B', 18)
    pdf.cell(0, 10, "MATRIX ENGINE // EXECUTIVE BRIEF", ln=True)
    pdf.set_draw_color(50, 50, 50)
    pdf.line(10, 22, 200, 22)
    pdf.ln(8)
    
    if shock_target:
        pdf.set_font("Courier", 'B', 10)
        pdf.set_text_color(200, 50, 50)
        pdf.cell(0, 6, f"[!] SCENARIO INJECTED: RATING PENALTY APPLIED TO {shock_target.upper()}", ln=True)
        pdf.ln(4)
        
    pdf.set_font("Courier", '', 11)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 6, "WINNER :", ln=True)
    pdf.set_font("Courier", 'B', 22)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, f"CHAMPION: {final_winner.upper()}", ln=True)
    pdf.ln(8)
    
    min_prob = 1.0
    risk_match = None
    for stage, matches in results_tree.items():
        for m in matches:
            if m['prob'] < min_prob:
                min_prob = m['prob']
                risk_match = m
                
    if risk_match:
        pdf.set_font("Courier", '', 11)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 6, "HIGHEST RISK NODE (TIGHTEST MARGIN):", ln=True)
        pdf.set_font("Courier", 'B', 12)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 8, f"{risk_match['team_a'].upper()} vs {risk_match['team_b'].upper()} -> {risk_match['winner'].upper()} ({risk_match['prob']*100:.1f}%)", ln=True)
        pdf.ln(8)

    pdf.set_font("Courier", 'B', 12)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 8, "STRUCTURAL KNOCKOUT PATH:", ln=True)
    
    for stage in ["Round of 16", "Quarter-Finals", "Semi-Finals", "World Cup Final"]:
        pdf.set_font("Courier", 'B', 10)
        pdf.set_text_color(200, 200, 200)
        pdf.cell(0, 8, f"--- {stage.upper()} ---", ln=True)
        pdf.set_font("Courier", '', 9)
        pdf.set_text_color(140, 140, 140)
        for m in results_tree.get(stage, []):
            pdf.cell(0, 5, f"[{m['prob']*100:.1f}%] {m['team_a'].upper()} vs {m['team_b'].upper()}  >>>  {m['winner'].upper()} ADVANCES", ln=True)
        pdf.ln(3)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        with open(tmp.name, "rb") as f:
            return f.read()

# --- SIDEBAR ARCHITECTURE ---
with st.sidebar:
    st.markdown("<h2 style='color:#F0F0F0; font-family:var(--font-mono); font-size:1.1rem; letter-spacing:1px;'>◼ MATRIX_ENGINE</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='border:none; height:1px; background-color:#222; margin: 1rem 0;' />", unsafe_allow_html=True)
    status_dot("live", "SYS_ACTIVE")
    status_dot("idle", "ML_INFERENCE_READY")
    
    section_divider("MODEL WEIGHTS")
    weights = [
        ("TALENT BASELINE (MEAN RTG)", 65.0),
        ("SQUAD ELASTICITY (VALUATION)", OPTIMAL_ELASTICITY * 100),
        ("HISTORICAL PEDIGREE (XGBOOST)", OPTIMAL_PEDIGREE * 100),
        ("RECENT TROPHY ALPHA", OPTIMAL_TROPHY * 100),
        ("SQUAD VOLATILITY (STD DEV)", OPTIMAL_UNIFORMITY * 100)
    ]
    
    for label, val in weights:
        st.markdown(f"""
        <div style="margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 0.65rem; color: var(--text-secondary); margin-bottom: 4px; letter-spacing: 0.5px;">
                <span>{label}</span><span style="color:var(--text-primary);">{val:.1f}%</span>
            </div>
            <div style="width: 100%; height: 3px; background: var(--bg-subtle); border-radius: 2px;">
                <div style="width: {val}%; height: 100%; background: var(--text-primary); border-radius: 2px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    section_divider("ANOMALY INJECTION")
    enable_shock = st.toggle("Enable Shock Scenario")
    shock_target, shock_penalty = None, 0.0
    if enable_shock:
        all_teams = sorted(list(nt_df['nationality'].unique()))
        shock_target = st.selectbox("Select Target Entity", all_teams, index=all_teams.index('France') if 'France' in all_teams else 0)
        shock_penalty = st.slider("CRISIS SEVERITY (RATING DROP)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
        
        if shock_penalty <= 2.5:
            crisis_desc, color = "Minor fatigue / Suspension", "#A0A0A0"
        elif shock_penalty <= 6.0:
            crisis_desc, color = "ACL Tear for Star Captain", "#E0B0FF"
        elif shock_penalty <= 8.5:
            crisis_desc, color = "Multiple injuries / Sacked", "#FF9900"
        else:
            crisis_desc, color = "Catastrophic squad collapse", "#CC3333"
            
        st.markdown(f"<div style='font-size:0.75rem; color:var(--text-muted); font-family:var(--font-mono); margin-top:-10px; margin-bottom:15px;'>↳ SCENARIO: <span style='color:{color}; font-weight:bold;'>{crisis_desc}</span></div>", unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 40px; color:#555; font-size:0.75rem; font-family:var(--font-mono);'>v4.1.0-enterprise<br>Rendering Ops</div>", unsafe_allow_html=True)

# --- MATH ENGINE ---
def calculate_match_probability(team1, team2, target=None, penalty=0.0):
    r1 = nt_df.loc[nt_df['nationality'] == team1, 'rating_mean'].values[0] if team1 in nt_df['nationality'].values else 70.0
    r2 = nt_df.loc[nt_df['nationality'] == team2, 'rating_mean'].values[0] if team2 in nt_df['nationality'].values else 70.0
    
    if team1 == target: r1 -= penalty
    if team2 == target: r2 -= penalty
    
    std1 = nt_df.loc[nt_df['nationality'] == team1, 'rating_std'].values[0] if team1 in nt_df['nationality'].values else 4.0
    std2 = nt_df.loc[nt_df['nationality'] == team2, 'rating_std'].values[0] if team2 in nt_df['nationality'].values else 4.0
    v1 = nt_df.loc[nt_df['nationality'] == team1, 'value_m'].values[0] if team1 in nt_df['nationality'].values else 5.0
    v2 = nt_df.loc[nt_df['nationality'] == team2, 'value_m'].values[0] if team2 in nt_df['nationality'].values else 5.0
    
    talent_baseline = 0.5 + ((r1 - r2) * 0.035)
    uniformity_modifier = (std2 - std1) * OPTIMAL_UNIFORMITY
    value_elasticity = ((1 / (1 + np.exp(-0.004 * (v1 - 800)))) - (1 / (1 + np.exp(-0.004 * (v2 - 800))))) * OPTIMAL_ELASTICITY
    trophy_modifier = (trophy_weights.get(team1, 0.0) - trophy_weights.get(team2, 0.0)) * (OPTIMAL_TROPHY / 0.05)
    
    elo1, elo2 = elo_ratings.get(team1, 1500), elo_ratings.get(team2, 1500)
    c1, c2 = le.transform([assign_cont(team1)])[0], le.transform([assign_cont(team2)])[0]
    xgb_preds = xgb_model.predict_proba(pd.DataFrame([[elo1, elo2, elo1 - elo2, c1, c2]], columns=features))[0]
    pedigree_modifier = ((xgb_preds[2] + (xgb_preds[1] * 0.5)) - 0.5) * OPTIMAL_PEDIGREE
    
    prob = np.clip(talent_baseline + uniformity_modifier + value_elasticity + trophy_modifier + pedigree_modifier, 0.02, 0.98)
    if prob > 0.54: prob = min(0.99, prob + 0.12)
    elif prob < 0.46: prob = max(0.01, prob - 0.12)
    
    return prob, {"r1": r1, "r2": r2, "v1": v1/1000, "v2": v2/1000, "std1": std1, "std2": std2, "elo1": elo1, "elo2": elo2}

# --- EXECUTION STATE ---
if "execution_triggered" not in st.session_state:
    st.session_state.execution_triggered = False

with col2:
    if st.button("EXECUTE ⌘E", type="primary", use_container_width=True):
        st.session_state.execution_triggered = True

section_divider("SYSTEM OUTPUT")

if st.session_state.execution_triggered:
    world_cup_2026_groups = {
        'A': ['Mexico', 'South Africa', 'South Korea', 'Czechia'], 'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
        'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'], 'D': ['USA', 'Paraguay', 'Australia', 'Turkey'], 
        'E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'], 'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
        'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'], 'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
        'I': ['France', 'Senegal', 'Iraq', 'Norway'], 'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
        'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'], 'L': ['England', 'Croatia', 'Ghana', 'Panama']
    }
    
    group_points = {g: {t: 0 for t in teams} for g, teams in world_cup_2026_groups.items()}
    for g, teams in world_cup_2026_groups.items():
        for i in range(len(teams)):
            for j in range(i+1, len(teams)):
                prob, _ = calculate_match_probability(teams[i], teams[j], shock_target, shock_penalty)
                if prob > 0.55: group_points[g][teams[i]] += 3
                elif prob < 0.45: group_points[g][teams[j]] += 3
                else:
                    group_points[g][teams[i]] += 1
                    group_points[g][teams[j]] += 1

    w, ru, thirds = {}, {}, []
    for g, teams in world_cup_2026_groups.items():
        ranked = sorted(group_points[g].items(), key=lambda x: (x[1], elo_ratings.get(x[0], 1500)), reverse=True)
        w[g], ru[g] = ranked[0][0], ranked[1][0]
        thirds.append({"team": ranked[2][0], "points": ranked[2][1]})

    thirds.sort(key=lambda x: (x['points'], elo_ratings.get(x['team'], 1500)), reverse=True)
    wc = [t['team'] for t in thirds[:8]]

    knockout_grid = [
        (ru['A'], ru['B']), (w['E'], wc[0]), (w['F'], ru['C']), (w['C'], ru['F']),
        (w['I'], wc[1]), (ru['E'], ru['I']), (w['A'], wc[2]), (w['L'], wc[3]),
        (w['D'], wc[4]), (w['G'], wc[5]), (ru['K'], ru['L']), (w['H'], ru['J']),
        (w['B'], wc[6]), (w['J'], ru['H']), (w['K'], wc[7]), (ru['D'], ru['G'])
    ]

    stages_list = ["Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals", "World Cup Final"]
    current_tier = knockout_grid
    results_tree = {}
    
    with st.status("EXECUTING LIKELIHOOD CONVERGENCE...", expanded=True) as status:
        for idx, stage_name in enumerate(stages_list):
            results_tree[stage_name] = []
            next_tier_teams = []
            for side_a, side_b in current_tier:
                prob_a, stats_dict = calculate_match_probability(side_a, side_b, shock_target, shock_penalty)
                victor = side_a if prob_a >= 0.50 else side_b
                next_tier_teams.append(victor)
                results_tree[stage_name].append({
                    "team_a": side_a, "team_b": side_b, "winner": victor, 
                    "prob": prob_a if victor == side_a else (1 - prob_a),
                    "stats": stats_dict
                })
            if len(next_tier_teams) > 1:
                current_tier = [(next_tier_teams[j], next_tier_teams[j+1]) for j in range(0, len(next_tier_teams), 2)]
        status.update(label="CONVERGENCE COMPLETE", state="complete")

    tab_g, tab_32, tab_16, tab_q, tab_s, tab_f = st.tabs(["STANDINGS", "R32", "R16", "QUARTERS", "SEMIS", "GRAND FINAL"])

    with tab_g:
        g_cols = st.columns(4, gap="large")
        for idx, (g_name, teams_pts) in enumerate(group_points.items()):
            with g_cols[idx % 4]:
                html = f"<div class='card' style='padding: 1rem;'><div class='card-label'>GROUP {g_name}</div>"
                for t, p in sorted(teams_pts.items(), key=lambda x: x[1], reverse=True):
                    intensity = p / 9.0  
                    bg_color = f"rgba(255, 255, 255, {intensity * 0.15})"
                    border_style = "border-left: 3px solid var(--accent);" if p >= max(teams_pts.values()) else "border-left: 3px solid transparent;"
                    html += f"<div style='margin-top:6px; padding: 6px 10px; background-color: {bg_color}; {border_style} border-radius: 4px; font-family:var(--font-mono); font-size:0.85rem; display:flex; justify-content:space-between; color:var(--text-primary);'><span>{get_code(t)} {t.upper()}</span> <span style='font-weight:700;'>{p}</span></div>"
                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)

    def draw_terminal_stage(stage_key, cols_layout=4):
        cols = st.columns(cols_layout, gap="medium")
        for idx, m in enumerate(results_tree[stage_key]):
            with cols[idx % cols_layout]:
                
                # HTML String is completely flattened to left margin to bypass Streamlit Markdown Parsing.
                html_card = f"""
<label class="flip-container" title="Click to inspect metrics">
<input type="checkbox" style="display:none;">
<div class="flipper">
<div class="front card">
<div class="card-label">NODE {idx+1:02d}</div>
<div style="font-family: var(--font-mono); font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 12px; line-height: 1.6;">
{get_code(m['team_a'])} {m['team_a'].upper()}<br>
<span style="color: var(--text-muted); font-size: 0.7rem;">VS</span><br>
{get_code(m['team_b'])} {m['team_b'].upper()}
</div>
<div style="border-top: 1px solid var(--bg-subtle); padding-top: 12px; position: absolute; bottom: 20px; width: calc(100% - 3rem);">
<div style="font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px;">Advancing</div>
<div style="font-size: 1.1rem; font-weight: 700; color: var(--text-primary); font-family: var(--font-mono);">
{get_code(m['winner'])} {m['winner'].upper()}
</div>
</div>
</div>
<div class="back card">
<div class="card-label" style="color:var(--text-primary); border-bottom: 1px solid var(--bg-subtle); padding-bottom: 5px; margin-bottom: 10px;">
{get_code(m['winner'])} WIN PROB: {m['prob']*100:.1f}%
</div>
<div style="display:flex; justify-content:space-between;">
<div style="width: 48%;">
<div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:4px; font-weight:bold;">{get_code(m['team_a'])}</div>
<div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-secondary); line-height:1.5;">
Val: €{m['stats']['v1']:.2f}B<br>
Rtg: {m['stats']['r1']:.1f}<br>
Vol: {m['stats']['std1']:.2f}σ<br>
Elo: {m['stats']['elo1']:.0f}
</div>
</div>
<div style="width: 48%;">
<div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:4px; font-weight:bold;">{get_code(m['team_b'])}</div>
<div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-secondary); line-height:1.5;">
Val: €{m['stats']['v2']:.2f}B<br>
Rtg: {m['stats']['r2']:.1f}<br>
Vol: {m['stats']['std2']:.2f}σ<br>
Elo: {m['stats']['elo2']:.0f}
</div>
</div>
</div>
<div style="position: absolute; bottom: 12px; width: 100%; text-align: center; font-size: 0.65rem; color: var(--text-muted); font-family: var(--font-mono);">
⟲ CLICK TO RETURN
</div>
</div>
</div>
</label>
"""
                st.markdown(html_card, unsafe_allow_html=True)

    with tab_32: draw_terminal_stage("Round of 32", 4)
    with tab_16: draw_terminal_stage("Round of 16", 4)
    with tab_q: draw_terminal_stage("Quarter-Finals", 2)
    with tab_s: draw_terminal_stage("Semi-Finals", 2)
    
    with tab_f:
        draw_terminal_stage("World Cup Final", 1)
        final_winner = results_tree["World Cup Final"][0]["winner"]
        
        st.markdown(f"""
            <div style="text-align: center; margin-top: 40px; padding: 60px; border: 1px solid var(--text-muted); background: var(--bg-surface); border-radius: var(--radius-lg); box-shadow: var(--shadow-card);">
                <div style="color:var(--text-muted); font-family:var(--font-mono); font-size:0.85rem; letter-spacing:0.2em; margin-bottom:1rem;">TOURNAMENT CONVERGENCE ACHIEVED</div>
                <h1 style="font-size: 4rem; color: var(--text-primary); margin: 0; font-family: var(--font-display); font-weight: 700; letter-spacing: -0.04em;">
                    {get_code(final_winner)} {final_winner.upper()}
                </h1>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        pdf_bytes = build_executive_pdf(results_tree, final_winner, shock_target if enable_shock else None)
        
        st.download_button(
            label="DOWNLOAD EXECUTIVE PDF BRIEF",
            data=pdf_bytes,
            file_name="matrix_briefing.pdf",
            mime="application/pdf",
            type="primary"
        )