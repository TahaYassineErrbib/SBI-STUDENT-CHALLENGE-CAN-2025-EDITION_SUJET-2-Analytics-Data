CAN 2025 Match Outcome Predictor – Streamlit App

This repository contains an interactive Streamlit dashboard developed for the SBI Student Challenge – CAN 2025 Edition (Analytics & Data track).

The application allows users to explore recent team performance and predict match outcomes between the 25 qualified teams for CAN 2025, using historical data and an interpretable predictive model.

Project Overview

The objective of this dashboard is to:

Analyze recent performance trends of CAN 2025 teams

Estimate match outcome probabilities (win / draw / loss)

Provide clear, data-driven insights through an interactive interface

All predictions are based on team-level historical performance over the last 5 years.

Application Features

Team selection with validation (no duplicate teams)

Team overview statistics:

Matches, wins, draws, losses

Win rate, draw rate

Goal difference

Average goals scored and conceded

Display of each team’s last 5 matches

Match outcome probabilities:

Team A win

Team B win

Draw

Compact probability bar chart

Recent performance trend visualization (last 5 matches)

Model Description

The predictive model is a binary logistic regression, trained to estimate the probability that one team wins against another based on:

Goal Difference Advantage

Win Rate Advantage

Draw Rate Tendency

Because CAN 2025 matches are played on neutral venues, the model does not assume a true home advantage.
The final output is adjusted using historical draw frequencies to provide three outcome probabilities:

Team A win

Team B win

Draw

This approach preserves model interpretability while producing realistic match predictions.

Repository Structure
.
├── app.py                # Streamlit application
├── team_stats.csv        # Team-level performance metrics
├── logreg_model.pkl      # Trained logistic regression model
├── results.csv           # Historical match results (last 5 years)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

How to Run the App Locally
1. Install dependencies
pip install -r requirements.txt

2. Launch the Streamlit app
streamlit run app.py


The application will open automatically in your web browser.

Live Demo

A live version of the application is available here:
[Insert Streamlit Cloud URL here]

No installation is required to access the live demo.

Data Sources

Publicly available international football match results

Data filtered to include only CAN 2025 qualified teams

Time window: last 5 years to reflect recent form

Notes for Reviewers

The model prioritizes clarity and interpretability over complexity

Predictions are probabilistic, not deterministic

Draw probability is handled explicitly to reflect tournament realities

The dashboard is designed for analysts, decision-makers, and non-technical users

Author

Name: [Your Name]
Institution: [Your University / School]
Challenge: SBI Student Challenge – CAN 2025 Edition
Track: Analytics & Data
