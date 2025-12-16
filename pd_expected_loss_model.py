
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ----------------------------
# Load borrower data
# ----------------------------
# Update filename if needed before running
df = pd.read_csv("loan_borrower_data.csv")

# Target variable
TARGET = "default"

# Feature set (drop identifiers and target)
X = df.drop(columns=[TARGET, "customer_id"], errors="ignore")
y = df[TARGET]

# ----------------------------
# Train PD model
# ----------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])

model.fit(X, y)

# ----------------------------
# PD estimation function
# ----------------------------
def predict_pd(loan_features: dict):
    input_df = pd.DataFrame([loan_features])
    pd_estimate = model.predict_proba(input_df)[0][1]
    return float(pd_estimate)

# ----------------------------
# Expected Loss function
# ----------------------------
def expected_loss(loan_features: dict, recovery_rate=0.10):
    pd_estimate = predict_pd(loan_features)
    exposure = loan_features.get("loan_amt_outstanding", 0)
    loss_given_default = 1 - recovery_rate
    return float(pd_estimate * exposure * loss_given_default)

# ----------------------------
# Sample test
# ----------------------------
if __name__ == "__main__":
    sample_loan = {
        "income": 75000,
        "credit_lines_outstanding": 4,
        "loan_amt_outstanding": 20000,
        "total_debt_outstanding": 35000,
        "years_employed": 6,
        "fico_score": 680
    }

    print("PD:", round(predict_pd(sample_loan), 4))
    print("Expected Loss:", round(expected_loss(sample_loan), 2))
