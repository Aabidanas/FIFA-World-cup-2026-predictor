# ⚽ THE MATRIX ENGINE
### FIFA 2026 World Cup Predictive Dashboard

> *"Anyone can predict the winner. This engine predicts what happens when everything breaks."*

A deterministic machine learning system that simulates the entire 2026 FIFA World Cup — and lets you stress-test it with real-world crisis scenarios like squad injuries, rating drops, and structural failures.

Built with a brutalist monochromatic UI. No fluff. Just data.

---

## 🔴 Live Demo
https://fifaworldcup2026predictor.streamlit.app/

---

## 🧠 What Makes This Different

Most predictors just rank teams. This one goes further.

| Feature | What It Does |
|---|---|
| **Shock Scenario Engine** | Inject a crisis (e.g. ACL tear for star captain) → watch the entire bracket recalculate live |
| **Full Bracket Simulation** | Group Stage → R32 → R16 → QF → SF → Grand Final |
| **Volatility Analysis** | Squads with high σ (inconsistency) get penalized — because knockout football punishes glass cannons |
| **Executive PDF Reports** | One-click auto-generated risk briefing for any matchup |
| **3D Match Cards** | Custom CSS flip-card engine showing Val, Elo, Rating, and σ side-by-side per match |

---

## ⚙️ How the Model Works

The probability engine runs a **5-factor weighted ensemble** on every match:

```
Win Probability = f(Talent, Valuation, History, Trophies, Volatility)
```

| Factor | Weight | What It Captures |
|---|---|---|
| FIFA Talent Baseline (Mean RTG) | **65%** | Raw squad quality |
| Squad Elasticity (Market Value) | **15%** | Depth and financial strength |
| Historical Pedigree (XGBoost) | **10%** | Win rates since 2006 |
| Recent Trophy Alpha | **8%** | Momentum from recent titles |
| Squad Volatility (Std Dev) | **2%** | Consistency — penalizes top-heavy squads |

> All weights are configurable in `app.py`.

---

## 🚀 Getting Started

**Requirements:** Python 3.8+

```bash
# 1. Clone the repo
git clone https://github.com/Aabidanas/FIFA-World-cup-2026-predictor
cd matrix-engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Make sure ea_fc26_players.csv is in the root folder
#    (required for squad elasticity calculations)

# 4. Run
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 📁 File Structure

```
matrix-engine/
│
├── app.py                    # Core engine: data ingestion, probability math, state
├── helper_ui.py              # Custom CSS, 3D flip-card mechanics, UI overrides
├── theme.py                  # Plotly themes and global color palette
├── requirements.txt          # All dependencies
├── ea_fc26_players.csv       # Player stats → squad elasticity + baseline ratings
├── ea_fc26_outfield.csv      # Outfield player metrics
└── ea_fc26_goalkeepers.csv   # Goalkeeper-specific data
```

---

## 🔬 Shock Scenario: How It Works

This is the engine's core differentiator.

1. Toggle **"Enable Shock Scenario"** in the sidebar
2. Select a **target team** (e.g. France)
3. Set **crisis severity** (rating drop slider)
4. The system maps this to a specific scenario (e.g. *ACL Tear for Star Captain*)
5. Every downstream match probability recalculates instantly

**Real example from the model:**
> Brazil vs England — Semifinal
> Squad values within €0.3B of each other. Elo within 21 points.
> England win probability: **52.7%**
> Deciding variable: England's squad σ was 0.14 higher than Brazil's.
> *In knockout stages, consistency beats talent.*

---

## 🛠️ Tech Stack

- **Framework:** Streamlit + custom CSS/HTML injection
- **ML Model:** XGBoost + Scikit-learn
- **Data:** Pandas, NumPy
- **Reporting:** FPDF (zero-dependency PDF compiler)
- **UI Engine:** Pure-CSS 3D flip cards (no JS libraries)

---

## 👤 Author

**Aabid Anas**
Data Science @ BITS Pilani Dubai × IIT Madras BS (Data Science)
[LinkedIn](#) · [GitHub](https://github.com/Aabidanas)

---

*Built for the 2026 FIFA World Cup. Open to feedback, forks, and heated arguments about your team's Elo rating.*
