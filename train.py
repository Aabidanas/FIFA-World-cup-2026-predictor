import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import json

OPTIMAL_DECAY = 0.15
url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
matches = pd.read_csv(url)
matches['date'] = pd.to_datetime(matches['date'])
modern_matches = matches[matches['date'] >= '2006-01-01'].copy()
modern_matches['outcome'] = np.where(modern_matches['home_score'] > modern_matches['away_score'], 2, np.where(modern_matches['home_score'] == modern_matches['away_score'], 1, 0))
max_date = modern_matches['date'].max()

elo_ratings, elo_shifts_tracker = {}, {}
home_elos, away_elos = np.zeros(len(modern_matches)), np.zeros(len(modern_matches))
home_arr, away_arr = modern_matches['home_team'].to_numpy(), modern_matches['away_team'].to_numpy()
outcome_arr, tournament_arr = modern_matches['outcome'].to_numpy(), modern_matches['tournament'].to_numpy()
date_arr = modern_matches['date'].to_numpy()

for i in range(len(modern_matches)):
    home, away, tourney = home_arr[i], away_arr[i], tournament_arr[i]
    if home not in elo_ratings: elo_ratings[home] = 1500
    if away not in elo_ratings: elo_ratings[away] = 1500
    if home not in elo_shifts_tracker: elo_shifts_tracker[home] = []
    if away not in elo_shifts_tracker: elo_shifts_tracker[away] = []
        
    home_elos[i], away_elos[i] = elo_ratings[home], elo_ratings[away]
    actual = 1.0 if outcome_arr[i] == 2 else 0.5 if outcome_arr[i] == 1 else 0.0
    expected = 1 / (1 + 10 ** ((elo_ratings[away] - elo_ratings[home]) / 400))
    base_k = 50 if tourney == 'FIFA World Cup' else 30 if any(x in tourney for x in ['Qualifiers', 'Euro', 'Copa']) else 10
    adaptive_k = base_k * np.exp(-OPTIMAL_DECAY * (max_date - pd.Timestamp(date_arr[i])).days / 365.25)
    
    elo_shift = adaptive_k * (actual - expected)
    elo_ratings[home] += elo_shift
    elo_ratings[away] -= elo_shift
    
    elo_shifts_tracker[home].append(elo_shift)
    elo_shifts_tracker[away].append(-elo_shift)
    
modern_matches['home_elo'], modern_matches['away_elo'] = home_elos, away_elos
modern_matches['elo_difference'] = modern_matches['home_elo'] - modern_matches['away_elo']

form_volatility = {t: np.std(s[-25:]) if len(s) > 0 else 10.0 for t, s in elo_shifts_tracker.items()}

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

model.save_model('xgb_model.json')
with open('elo_ratings.json', 'w') as f: json.dump(elo_ratings, f)
with open('form_volatility.json', 'w') as f: json.dump(form_volatility, f)