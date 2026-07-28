"""
Generates a synthetic loan-default dataset for demo/training purposes.
Replace with a real dataset (e.g. LendingClub, Kaggle Home Credit) for
production use — this exists so the repo is runnable end-to-end offline.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000

age = np.random.randint(21, 65, n)
income = np.random.normal(55000, 20000, n).clip(15000, 200000)
loan_amount = np.random.normal(15000, 8000, n).clip(1000, 60000)
credit_score = np.random.normal(650, 80, n).clip(300, 850)
employment_years = np.random.exponential(5, n).clip(0, 40)
existing_debt = np.random.normal(8000, 6000, n).clip(0, 50000)
num_credit_lines = np.random.randint(1, 15, n)

debt_to_income = (existing_debt + loan_amount) / income
default_logit = (
    -6
    + 4.5 * debt_to_income
    - 0.01 * (credit_score - 650)
    - 0.05 * employment_years
    + 0.00002 * loan_amount
)
prob_default = 1 / (1 + np.exp(-default_logit))
default = np.random.binomial(1, prob_default)

df = pd.DataFrame(
    {
        "age": age,
        "income": income.round(2),
        "loan_amount": loan_amount.round(2),
        "credit_score": credit_score.round(0),
        "employment_years": employment_years.round(1),
        "existing_debt": existing_debt.round(2),
        "num_credit_lines": num_credit_lines,
        "default": default,
    }
)

df.to_csv("data/loan_data.csv", index=False)
print(f"Generated {len(df)} rows. Default rate: {df['default'].mean():.2%}")
