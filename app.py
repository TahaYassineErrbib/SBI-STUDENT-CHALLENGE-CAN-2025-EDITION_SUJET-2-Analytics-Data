# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load data and model
# -----------------------------
team_stats_df = pd.read_csv("team_stats.csv", index_col=0)
with open("logreg_model.pkl", "rb") as f:
    model = pickle.load(f)
df = pd.read_csv("results.csv", parse_dates=['date'])

# Keep last 5 years
recent_cutoff = pd.Timestamp.now() - pd.DateOffset(years=5)
df = df[df['date'] >= recent_cutoff]

AFCON_TEAMS = team_stats_df.index.tolist()

# -----------------------------
# App layout
# -----------------------------
st.set_page_config(
    page_title="CAN 2025 Match Outcome Predictor – SBI Student Challenge",
    layout="wide"
)

st.title("CAN 2025 Match Outcome Predictor")
st.markdown(
    """
    **Data-driven match outcome probabilities for the Africa Cup of Nations 2025.**  
    Predictions are based on recent international performance (last 5 years) of qualified teams.
    """
)

# -----------------------------
# Team selection
# -----------------------------
st.subheader("Match Configuration")

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox(
        "Team A (Reference Team)",
        AFCON_TEAMS,
        help="Team A is treated as the reference team in the prediction model."
    )
with col2:
    away_team = st.selectbox(
        "Team B (Opponent)",
        AFCON_TEAMS,
        help="Predictions are asymmetric: probabilities are computed relative to Team A."
    )

if home_team == away_team:
    st.warning("Please select two different teams to simulate a match.")
    st.stop()

# -----------------------------
# Team overview statistics
# -----------------------------
st.subheader("Team Performance Overview (Last 5 Years)")

home_stats = team_stats_df.loc[home_team]
away_stats = team_stats_df.loc[away_team]

stats_df = pd.DataFrame({
    home_team: home_stats,
    away_team: away_stats
}).T

st.dataframe(
    stats_df[
        [
            'matches','wins','draws','losses',
            'win_rate','draw_rate',
            'goal_difference',
            'avg_goals_scored','avg_goals_conceded'
        ]
    ],
    use_container_width=True
)

# -----------------------------
# Recent matches
# -----------------------------
st.subheader("Recent Match History")

def last_matches(team):
    matches = df[
        (df['home_team'] == team) | (df['away_team'] == team)
    ].sort_values('date', ascending=False).head(5)
    return matches[['date','home_team','home_score','away_team','away_score']]

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**{home_team} – Last 5 Matches**")
    st.dataframe(last_matches(home_team), use_container_width=True)
with col2:
    st.markdown(f"**{away_team} – Last 5 Matches**")
    st.dataframe(last_matches(away_team), use_container_width=True)

# -----------------------------
# Match outcome prediction
# -----------------------------
st.subheader("Predicted Match Outcome")

def predict_match(team, opponent):
    t_stats = team_stats_df.loc[team]
    o_stats = team_stats_df.loc[opponent]

    goal_diff_adv = (t_stats['avg_goals_scored'] - o_stats['avg_goals_conceded']) - \
                    (o_stats['avg_goals_scored'] - t_stats['avg_goals_conceded'])
    win_rate_adv = t_stats['win_rate'] - o_stats['win_rate']
    draw_rate_adv = (t_stats['draws']/t_stats['matches']) - \
                    (o_stats['draws']/o_stats['matches'])

    features = pd.DataFrame(
        [[goal_diff_adv, win_rate_adv, draw_rate_adv]],
        columns=['goal_diff_adv','win_rate_adv','draw_rate_adv']
    )

    # Binary logistic regression: Team A win vs not Team A win
    prob_team_win = model.predict_proba(features)[0][1]
    prob_team_not_win = 1 - prob_team_win

    # Historical draw probability adjustment
    total_games = t_stats['matches'] + o_stats['matches']
    draw_prob = (t_stats['draws'] + o_stats['draws']) / total_games
    draw_prob = np.clip(draw_prob, 0, 1)

    prob_team_win *= (1 - draw_prob)
    prob_opponent_win = prob_team_not_win * (1 - draw_prob)

    if abs(prob_team_win - prob_opponent_win) < 0.05:
        predicted = "High uncertainty (balanced match)"
    else:
        predicted = team if prob_team_win > prob_opponent_win else opponent

    return prob_team_win, prob_opponent_win, draw_prob, predicted

prob_home, prob_away, draw_prob, predicted = predict_match(home_team, away_team)

st.markdown(
    f"""
    **Model Verdict:** {predicted}  
    Based on recent performance metrics, **{home_team} has a {prob_home:.0%} probability of winning** this match.
    """
)

col1, col2, col3 = st.columns(3)
col1.metric("Team A Win Probability", f"{prob_home:.1%}")
col2.metric("Draw Probability", f"{draw_prob:.1%}")
col3.metric("Team B Win Probability", f"{prob_away:.1%}")

# -----------------------------
# Probability distribution chart
# -----------------------------
st.subheader("Predicted Outcome Distribution")

prob_df = pd.DataFrame({
    'Outcome': [f"{home_team} Win", "Draw", f"{away_team} Win"],
    'Probability': [prob_home, draw_prob, prob_away]
})

fig, ax = plt.subplots(figsize=(5,3))
sns.barplot(
    x='Outcome',
    y='Probability',
    data=prob_df,
    palette='Blues',
    ax=ax
)
ax.set_ylim(0, 1)
ax.set_ylabel("Probability")
ax.set_xlabel("")
for i, v in enumerate(prob_df['Probability']):
    ax.text(i, v + 0.02, f"{v:.1%}", ha='center', fontsize=9)

st.pyplot(fig, use_container_width=False)

# -----------------------------
# Recent performance trends
# -----------------------------
st.subheader("Short-Term Performance Trend (Last 5 Matches)")

def team_trend(team):
    last5 = df[
        (df['home_team'] == team) | (df['away_team'] == team)
    ].sort_values('date', ascending=False).head(5)

    results = []
    for _, row in last5.iterrows():
        if row['home_team'] == team:
            if row['home_score'] > row['away_score']:
                results.append('Win')
            elif row['home_score'] == row['away_score']:
                results.append('Draw')
            else:
                results.append('Loss')
        else:
            if row['away_score'] > row['home_score']:
                results.append('Win')
            elif row['away_score'] == row['home_score']:
                results.append('Draw')
            else:
                results.append('Loss')

    return results[::-1], last5['date'].dt.strftime('%Y-%m-%d').tolist()[::-1]

home_results, home_dates = team_trend(home_team)
away_results, away_dates = team_trend(away_team)

fig, ax = plt.subplots(figsize=(6,2.5))
ax.plot(
    home_dates,
    [1 if r == 'Win' else 0 if r == 'Draw' else -1 for r in home_results],
    marker='o',
    label=home_team
)
ax.plot(
    away_dates,
    [1 if r == 'Win' else 0 if r == 'Draw' else -1 for r in away_results],
    marker='o',
    label=away_team
)

ax.set_yticks([-1, 0, 1])
ax.set_yticklabels(['Loss', 'Draw', 'Win'])
ax.set_xlabel("Match Date")
ax.set_ylabel("Result")
ax.set_title("Recent Match Momentum Indicator")
ax.grid(True, alpha=0.3)
ax.legend()
plt.xticks(rotation=45, ha='right')

st.pyplot(fig, use_container_width=False)

# -----------------------------
# Interpretability & disclaimer
# -----------------------------
with st.expander("How to interpret these results"):
    st.markdown(
        """
        - Probabilities are estimated using a **logistic regression model** trained on international CAF matches from the last 5 years.
        - Features capture **relative team strength**, including win rate, goal difference, and draw tendency.
        - The model focuses on **historical performance patterns** and does not account for injuries, lineups, or tactical choices.
        - Outputs are **probabilistic**, not deterministic predictions.
        """
    )

st.caption(
    "SBI Student Challenge 2025 – Analytics & Data Track | CAN 2025 Match Outcome Predictor"
)
