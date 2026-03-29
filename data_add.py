import pandas as pd
import numpy as np

df = pd.read_csv("traffic-prediction-dataset.csv")

# Fix column spacing issue
df.columns = df.columns.str.strip()

# Calculate city average traffic
df["City_Average_Traffic"] = df[
    ["Cross 1","Cross 2","Cross 3","Cross 4","Cross 5","Cross 6"]
].mean(axis=1)

# Create time column
df["Time"] = pd.date_range(start="2024-01-01", periods=len(df), freq="5min")

# Extract hour
df["Hour"] = df["Time"].dt.hour

# Peak hour indicator
df["Peak"] = df["Hour"].apply(lambda x: 1 if (8<=x<=10 or 17<=x<=19) else 0)

# Baseline traffic
baseline = df["City_Average_Traffic"].mean()

# Congestion index
df["Congestion_Index"] = df["City_Average_Traffic"] / baseline

# Congestion level
def congestion(x):
    if x < 1:
        return "Low"
    elif x < 1.3:
        return "Medium"
    else:
        return "High"

df["Congestion_Level"] = df["Congestion_Index"].apply(congestion)

# Save enhanced dataset
df.to_csv("enhanced_traffic_dataset.csv", index=False)

print("Enhanced dataset created successfully")
