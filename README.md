# MSc Dissertation Project — Interpretable Stellar Mass Estimation from Photometric Data

*Interpretable machine learning model that recovers a physical law for galaxy stellar mass using only photometric data.*

## Overview

This project focuses on estimating **galaxy stellar mass** using only **photometric observations** (no spectroscopy), combining:

- Machine Learning (Random Forest)
- Symbolic Regression (PySR)

The goal is to move from **black-box prediction → interpretable analytical formula**, making the model both accurate and usable in real-world scenarios.

## Problem Statement

Estimating stellar mass is a fundamental task in astrophysics, but traditional methods:

- Require expensive **SED fitting**
- Depend on strong assumptions (stellar population, dust, etc.)
- Do not scale well to large surveys

👉 This project answers:

> Can we estimate stellar mass **accurately + efficiently + interpretably** using only photometric data?

## Methodology

### 1. Data

Mock galaxy catalog simulating real surveys

Features:

- Redshift (z)
- ugriz photometry
- Derived absolute magnitudes
- Color indices (e.g., i − z)

Flux-limited selection:

- `m_i < 22.5` (ensures data completeness)

### 2.Feature Engineering

Constructed physically meaningful features:

- Absolute magnitudes: $M_u, M_g, M_r, M_i, M_z$
- Colors:
  - `u-g`, `g-r`, `r-i`, `i-z`

Key insight:

- Color = proxy for stellar population
- Red bands = better mass tracers

### 3.Modeling Pipeline

#### (A) Random Forest (Feature Discovery)

- Model: `sklearn RandomForestRegressor`
- Purpose:
  - Capture nonlinear relationships
  - Identify key predictors

 **Result:**

- Top features (low redshift):

  * `M_z` → dominant (~79%)

  - `i - z` → second (~16%)

👉 Together explain **>95% of predictive power** 

#### (B) Symbolic Regression (Model Simplification)

- Tool: `PySR`
- Goal:
  - Extract **explicit analytical formula**
  - Balance accuracy & interpretability

## Key Results

### Final Model (Interpretable Formula)

For low redshift galaxies (z < 0.5):

```
log10(M) = -0.335 * M_z + 3.527 * (i - z) + 2.455
```

📌 Meaning:

- `M_z` → total stellar light (luminosity)
- `i - z` → stellar population / age indicator
- Constant → normalization

### Performance

| Model               | R²    | RMSE      |
| ------------------- | ----- | --------- |
| Random Forest       | ~0.99 | ~0.07 dex |
| Symbolic Regression | ~0.96 | ~0.15 dex |

👉 Trade-off:

- RF = higher accuracy
- SR = slightly lower accuracy but **fully interpretable**

## Key Insights

### 1.Simple Physics Behind Complex Data

Despite high-dimensional inputs:

> Stellar mass is almost fully determined by just:

- z-band magnitude
- i−z color

### 2.Redshift Matters

* Best performance: **z < 0.5**

* Higher redshift → more noise & weaker relation

### 3.Interpretability vs Performance

* Black-box models ≠ always better

* Simple formulas can achieve **near-equivalent performance**

## Project Structure

├── figures/        # Plots and visualizations
├── src/            # Modeling & analysis code
├── thesis/         # Full dissertation (PDF)
└── README.md

## Tech Stack

* Python

* NumPy / Pandas

* Scikit-learn

* PySR (Symbolic Regression)

* Matplotlib / Seaborn

## Use Cases

This workflow is applicable beyond astrophysics:

- Feature selection & dimensionality reduction
- Interpretable ML for scientific data
- Deriving analytical formulas from data
- Replacing black-box models in production

## What I Can Offer (Freelance)

I can help with:

- Data analysis & EDA
- Feature engineering for structured data
- Time series / tabular ML modeling
- Interpretable ML (symbolic regression)
- Turning research into production-ready pipelines
- Writing clear technical reports (EN/CN)

## Contact

Feel free to reach out for collaboration or freelance work.

My email: cyujie0228@163.com

Upwork:[Chen Y. - Data Analyst | Turning Data into Insights & Predictive Models - Upwork Freelancer from Chengdu, China](https://www.upwork.com/freelancers/~019535ac95f3eab6c7)
