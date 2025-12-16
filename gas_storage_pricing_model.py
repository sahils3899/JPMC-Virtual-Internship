
import pandas as pd
import numpy as np

# ----------------------------
# Load and prepare price data
# ----------------------------
df = pd.read_csv("Nat_Gas.csv")
df.columns = ["Date", "Price"]
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)

daily_prices = df.resample("D").interpolate(method="linear")

# ----------------------------
# Gas price estimator
# ----------------------------
def estimate_gas_price(date_input):
    date_input = pd.to_datetime(date_input)
    return float(daily_prices.loc[date_input, "Price"])

# ----------------------------
# Storage contract pricing
# ----------------------------
def price_storage_contract(
    injection_dates,
    withdrawal_dates,
    injection_rate,
    withdrawal_rate,
    max_volume,
    storage_cost_per_day
):
    inventory = 0.0
    contract_value = 0.0

    all_dates = sorted(set(injection_dates + withdrawal_dates))

    for date in all_dates:
        price = estimate_gas_price(date)

        if date in injection_dates:
            inject_volume = min(injection_rate, max_volume - inventory)
            inventory += inject_volume
            contract_value -= inject_volume * price

        if date in withdrawal_dates:
            withdraw_volume = min(withdrawal_rate, inventory)
            inventory -= withdraw_volume
            contract_value += withdraw_volume * price

        contract_value -= storage_cost_per_day

    return contract_value

# ----------------------------
# Sample test
# ----------------------------
if __name__ == "__main__":
    injection_dates = [
        "2023-01-31",
        "2023-02-28",
        "2023-03-31"
    ]

    withdrawal_dates = [
        "2023-10-31",
        "2023-11-30",
        "2023-12-31"
    ]

    value = price_storage_contract(
        injection_dates=injection_dates,
        withdrawal_dates=withdrawal_dates,
        injection_rate=10000,
        withdrawal_rate=10000,
        max_volume=30000,
        storage_cost_per_day=500
    )

    print("Contract Value:", round(value, 2))
