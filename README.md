
# CAN 2025 Match Outcome Predictor

This repository contains a **Streamlit dashboard** developed for the **SBI Student Challenge 2025 (Analytics & Data Track)**.  
The application predicts match outcomes for the **Africa Cup of Nations (CAN) 2025** using recent team performance data and a logistic regression model.

---

## Project Overview

The goal of this project is to provide **data-driven match outcome probabilities** to support:

- Sports analytics
- Tournament previewing
- Decision-making and storytelling around team performance

The model is trained on **international match data from the last 5 years** and focuses on **team-level performance metrics**.  
Home and away context is intentionally de-emphasized due to its limited relevance in CAN tournaments.

---

## Features

- Team selection with duplicate prevention
- Team performance overview
- Last 5 matches trend visualization
- Win / Draw / Loss probability estimation
- Compact charts optimized for demos
- Fully reproducible pipeline

---

## Model Summary

- **Model type:** Logistic Regression  
- **Target:** Match outcome (Win vs Not Win)  
- **Draw handling:**  
  Draw probability is inferred using calibrated score differences and historical draw frequency.  
- **Why Logistic Regression?**
  - Interpretable coefficients for each feature
  - Stable with limited historical data
  - Business-friendly explanations for stakeholders

---

## Repository Structure

```text
.
├── app.py                 # Streamlit app
├── requirements.txt       # Python dependencies
├── logreg_model.pkl       # Trained logistic regression model
├── results.cvs            # Main database used International football results from 1872 to 2025
├── team_stats.csv         # Team-level stats
├── notebook.ipynb         # Colab notebook for EDA and modeling
└── README.md              # This file
```
## How to Run the Streamlit App Locally

### 1. Clone the repository

```bash
git clone https://github.com/TahaYassineErrbib/SBI-STUDENT-CHALLENGE-CAN-2025-EDITION_SUJET-2-Analytics-Data
cd your-repo-name
```
### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Linux / Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the app
```bash
streamlit run app.py
```
The app will open in your browser at http://localhost:8501

Live app (optional): https://can2025-match-predictor.streamlit.app
