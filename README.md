# 💼 Job Salary Analysis — End-to-End Data Science Project

> **Best Model:** Gradient Boosting · Test R² = **0.8975** · Test MAE = **$7,693**  
> **Dataset:** `https://www.kaggle.com/datasets/mohithsairamreddy/salary-data` · 6,698 clean records · 191 unique job titles

---

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset Summary](#dataset-summary)
- [Project Structure](#project-structure)
- [Feature Engineering](#feature-engineering)
- [Model Results](#model-results)
- [Key Findings](#key-findings)
- [Setup & Execution](#setup--execution)
- [Deliverables](#deliverables)
- [Requirements](#requirements)

---

## Problem Statement

Understanding what drives compensation is valuable for job seekers, HR teams, and organizations setting pay structures. This project builds a **reproducible, end-to-end data science pipeline** that:

1. Explores and cleans a real-world salary dataset (6,698 records after cleaning).
2. Engineers 24 informative features from raw categorical and numeric columns.
3. Trains and compares **four regression models** with 5-fold cross-validation.
4. Produces **11 publication-ready visualizations** covering distributions, correlations, and model diagnostics.
5. Compiles all results into a formatted **Word report** (`Job_Salary_Analysis_Report.docx`).

The target variable is **Salary (USD)**, modelled on the log scale (`log1p`) to reduce right-skew and back-transformed for all reported error metrics.

---

## Dataset Summary

| Property | Value |
|---|---|
| **File** | `Salary_Data.csv` |
| **Raw rows** | 6,704 |
| **Clean rows** | 6,698 (6 NaN rows dropped) |
| **Raw features** | 5 input columns + 1 target |
| **Unique job titles** | 191 |
| **Salary range** | $350 – $250,000 |
| **Median salary** | $115,000 |

### Raw Columns

| Column | Type | Description |
|---|---|---|
| `Age` | Numeric | Employee age (21 – 62 years) |
| `Gender` | Categorical | Male / Female / Other |
| `Education Level` | Categorical | 7 variants: High School → Bachelor's → Master's → PhD |
| `Job Title` | Categorical | 191 unique titles |
| `Years of Experience` | Numeric | 0 – 34 years |
| `Salary` | Numeric *(target)* | USD · median $115,000 · max $250,000 |

---

## Project Structure

```
salary-prediction/
├── Salary_Data.csv                  # Raw dataset
├── salary_analysis.ipynb            # ★ Jupyter Notebook (5 cells, inline plots)
├── analysis.py                      # Full pipeline script
├── generate_docx_report.py          # Generates Salary_Project_Report.docx
├── requirements.txt                 # ★ Pinned dependencies
├── README.md                        # ★ This file
├── Job_Salary_Analysis_Report.docx  # ★ Word submission report
└── output_plots/                    # 11 saved chart images
    ├── 01_salary_distribution.png
    ├── 02_salary_by_education.png
    ├── 03_salary_by_gender.png
    ├── 04_experience_vs_salary.png
    ├── 05_top15_jobs_salary.png
    ├── 06_correlation_heatmap.png
    ├── 07_salary_by_exp_band.png
    ├── 08_senior_vs_nonsenior.png
    ├── 09_feature_importances.png
    ├── 10_actual_vs_predicted.png
    └── 11_residuals.png
```

---

## Feature Engineering

| Feature | Method | Rationale |
|---|---|---|
| `education_level_clean` | Fuzzy ordinal mapping (1–4) | Handles all 7 text variants (e.g. *"Bachelor's Degree"*, *"PhD"*) |
| `gender_encoded` | `LabelEncoder` → 0 / 1 / 2 | Converts Male / Female / Other |
| `is_senior` | Keyword flag (1/0) | Detects *senior, director, manager, vp, chief, head, lead, principal* |
| `jobd_*` (16 cols) | One-hot dummies | Top-15 most frequent job titles + "Other" bucket |
| `exp_x_edu` | `years_of_experience × education_level_clean` | Interaction: tenure × qualification |
| `age_squared` | `age²` | Non-linear age–salary curve |
| `exp_squared` | `years_of_experience²` | Diminishing returns of experience |
| `log_salary` | `log1p(salary)` | Normalises right-skewed target for linear models |

**Final feature matrix:** 6,698 rows × 24 features  
**Train / Test split:** 80% / 20% (`random_state=42`)

---

## Model Results

All models evaluated with **5-fold cross-validation** on the training set (5,358 samples), then scored on the held-out test set (1,340 samples). Errors back-transformed to USD.

| Model | CV R² (mean) | CV R² (±std) | Test R² | Test MAE | Test RMSE |
|---|---|---|---|---|---|
| Linear Regression | 0.8566 | ±0.0374 | 0.8498 | $16,186 | $22,388 |
| Ridge Regression (α=10) | 0.8547 | ±0.0370 | 0.8495 | $16,495 | $22,486 |
| Random Forest | 0.9201 | ±0.0410 | 0.8606 | $5,716 | $10,514 |
| **Gradient Boosting** ⭐ | **0.9172** | **±0.0389** | **0.8975** | **$7,693** | **$11,426** |

**Winner: Gradient Boosting** (`n_estimators=200`, `learning_rate=0.05`, `max_depth=5`)

---

## Key Findings

1. **Test R² = 0.8975** — Gradient Boosting explains ~90% of salary variance on unseen data.
2. **Experience dominates.** `years_of_experience` and `exp_squared` are the top two feature importances, confirming a non-linear tenure–salary relationship.
3. **Education × experience interaction matters.** The `exp_x_edu` term ranks highly — a PhD with 15 years earns disproportionately more than either factor alone predicts.
4. **Senior roles command a ~$50k premium.** The `is_senior` flag splits salaries into two distinct clusters: non-senior < $120k, senior/director/VP > $150k.
5. **Gender pay gap present.** Violin plot shows Male and Female medians differ by ~$10k–$15k after controlling for other factors.
6. **Linear models plateau at R² ≈ 0.85.** The jump to ensembles confirms significant non-linear salary structure.
7. **Residuals are well-behaved.** Approximately centred around zero with no strong heteroscedasticity, validating the log transformation.

---

## Setup & Execution

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2a — Run the Jupyter Notebook (recommended for submission)

```bash
jupyter notebook salary_analysis.ipynb
```

Then run all cells: **Kernel → Restart & Run All**

### 2b — Run the Python script

```bash
# Windows (UTF-8 console)
python -X utf8 analysis.py

# macOS / Linux
python analysis.py
```

### 3 — Generate the Word report

```bash
python generate_docx_report.py
```

> **Expected runtime:** ~60–120 seconds for the full model training (200-estimator ensembles).  
> **Python version:** 3.10+ recommended.

---

## Deliverables

| # | File | Description |
|---|---|---|
| 1 | `salary_analysis.ipynb` | Jupyter Notebook — 5 cells, inline plots, benchmark table |
| 2 | `requirements.txt` | All pinned project dependencies |
| 3 | `Job_Salary_Analysis_Report.docx` | Word report — abstract, methodology, tables, 11 charts |
| 4 | `README.md` | This file — overview, findings, setup instructions |

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
python-docx>=1.1.0
jupyter>=1.0.0
notebook>=7.0.0
ipykernel>=6.0.0
```

> **Python version:** 3.10 or later (uses built-in `list[str]` and `dict[K, V]` generics).

---

*Built with Python · pandas · scikit-learn · matplotlib · seaborn · python-docx*
