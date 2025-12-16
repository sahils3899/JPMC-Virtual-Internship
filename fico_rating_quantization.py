
import pandas as pd
import numpy as np

# ----------------------------
# Load borrower data
# ----------------------------
# Update filename if needed
df = pd.read_csv("loan_borrower_data.csv")

# Required columns
FICO_COL = "fico_score"
DEFAULT_COL = "default"

# ----------------------------
# Quantization using log-likelihood
# ----------------------------
def build_fico_rating_map(df, n_buckets=5):
    df = df.sort_values(FICO_COL).reset_index(drop=True)

    fico = df[FICO_COL].values
    default = df[DEFAULT_COL].values
    n = len(df)

    # Prefix sums for fast range queries
    prefix_defaults = np.cumsum(default)
    prefix_count = np.arange(1, n + 1)

    def bucket_log_likelihood(i, j):
        # Bucket from i to j inclusive
        ni = j - i + 1
        ki = prefix_defaults[j] - (prefix_defaults[i - 1] if i > 0 else 0)

        if ki == 0 or ki == ni:
            return 0.0

        pi = ki / ni
        return ki * np.log(pi) + (ni - ki) * np.log(1 - pi)

    # DP tables
    dp = np.full((n_buckets + 1, n), -np.inf)
    split = np.zeros((n_buckets + 1, n), dtype=int)

    # Base case
    for j in range(n):
        dp[1][j] = bucket_log_likelihood(0, j)

    # Fill DP
    for b in range(2, n_buckets + 1):
        for j in range(b - 1, n):
            for i in range(b - 1, j + 1):
                val = dp[b - 1][i - 1] + bucket_log_likelihood(i, j)
                if val > dp[b][j]:
                    dp[b][j] = val
                    split[b][j] = i

    # Recover boundaries
    boundaries = []
    j = n - 1
    for b in range(n_buckets, 0, -1):
        i = split[b][j]
        boundaries.append((fico[i], fico[j]))
        j = i - 1

    boundaries.reverse()

    rating_map = {}
    for idx, (low, high) in enumerate(boundaries, start=1):
        rating_map[idx] = {"min_fico": low, "max_fico": high}

    return rating_map

# ----------------------------
# Rating assignment function
# ----------------------------
def fico_to_rating(fico_score, rating_map):
    for rating, bounds in rating_map.items():
        if bounds["min_fico"] <= fico_score <= bounds["max_fico"]:
            return rating
    return max(rating_map.keys())

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    rating_map = build_fico_rating_map(df, n_buckets=5)
    print("Rating Map:", rating_map)
    print("Sample Rating:", fico_to_rating(720, rating_map))
