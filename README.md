# Hospital Readmission Analysis

Predicting 30-day hospital readmission risk in diabetic patients using the UCI
"Diabetes 130-US Hospitals" dataset (1999–2008, 101,766 encounters across 130 hospitals).

## Problem
Hospital readmissions within 30 days are costly and, under CMS policy, can trigger
financial penalties for hospitals. This project identifies the strongest clinical and
administrative predictors of 30-day readmission and builds an interpretable risk model
that could support discharge-planning decisions.

## Results
| Model | AUC-ROC |
|---|---|
| Logistic Regression (+SMOTE) | ~0.65 |
| Random Forest (+SMOTE) | ~0.64 |
| **XGBoost (class-weighted)** | **~0.67 (best)** |

**Top risk factors** (via SHAP): number of prior inpatient admissions, discharge
disposition, number of medications/diagnoses (comorbidity burden), and primary diagnosis
category (circulatory/diabetes-related admissions carry higher risk).

## Project Structure
```
readmission_project/
├── data/
│   └── diabetic_data.csv          # UCI Diabetes 130-Hospitals dataset
├── notebooks/
│   └── hospital_readmission_analysis.ipynb   # Full analysis, executed with outputs
├── outputs/                        # Saved chart images
├── src/
│   ├── clean.py                    # Data cleaning & feature engineering
│   └── model.py                    # Modeling pipeline (standalone script)
└── README.md
```

## Methodology
1. **Data Cleaning**: handled `?` placeholder missing values, dropped near-empty columns
   (`weight`, `payer_code`), collapsed ICD-9 diagnosis codes into clinical categories,
   excluded expired/hospice discharges, simplified the 3-class target to binary
   (`readmitted <30 days` vs. not).
2. **EDA**: readmission rate breakdowns by age, prior utilization, diagnosis category,
   discharge disposition, and medication count; correlation analysis.
3. **Statistical Testing**: chi-square tests (categorical features) and t-tests (numeric
   features) against the target.
4. **Modeling**: Logistic Regression, Random Forest, and XGBoost, with SMOTE / class
   weighting to handle the ~11% positive class imbalance. Evaluated on AUC-ROC (not
   accuracy, which is misleading on imbalanced data).
5. **Explainability**: SHAP values on the best model (XGBoost) to identify and visualize
   the top global drivers of readmission risk.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap jupyter
jupyter nbconvert --to notebook --execute --inplace notebooks/hospital_readmission_analysis.ipynb
```
Or open `notebooks/hospital_readmission_analysis.ipynb` directly in Jupyter — it already
contains all executed outputs and charts.

## Limitations
- AUC ~0.67 is consistent with published benchmarks on this dataset; 30-day readmission
  is a genuinely hard prediction problem, since a lot of the real drivers (social support,
  medication adherence, access to follow-up care) aren't captured in structured EHR data.
- Data is from 1999–2008; a production model would need retraining on more recent data.

## Dataset Source
Strack, B., et al. (2014). *Diabetes 130-US hospitals for years 1999-2008*.
UCI Machine Learning Repository.
