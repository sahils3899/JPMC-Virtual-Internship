
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("Nat_Gas.csv")
df.columns = ["Date", "Price"]
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)

# Convert monthly data to daily via interpolation
daily_prices = df.resample("D").interpolate(method="linear")

# Add time index and month for modeling
daily_prices["TimeIndex"] = np.arange(len(daily_prices))
daily_prices["Month"] = daily_prices.index.month

# Fit linear trend
trend_coef = np.polyfit(
    daily_prices["TimeIndex"], daily_prices["Price"], 1
)

# Monthly seasonality
monthly_seasonality = (
    daily_prices.groupby("Month")["Price"].mean()
    - daily_prices["Price"].mean()
)

# Pricing function
def estimate_gas_price(date_input):
    date_input = pd.to_datetime(date_input)

    if date_input <= daily_prices.index.max():
        return float(daily_prices.loc[date_input, "Price"])

    days_forward = (date_input - daily_prices.index.max()).days
    future_index = daily_prices["TimeIndex"].iloc[-1] + days_forward

    trend_price = trend_coef[0] * future_index + trend_coef[1]
    seasonal_adj = monthly_seasonality[date_input.month]

    return float(trend_price + seasonal_adj)


# Example usage
if __name__ == "__main__":
    print(estimate_gas_price("2022-05-15"))
    print(estimate_gas_price("2025-08-31"))
