import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "tourism.csv"
EXPECTED = ["CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"]
df = pd.read_csv(RAW_PATH)
missing = [c for c in EXPECTED if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")
if "ProdTaken" not in df or df["ProdTaken"].isna().any():
    raise ValueError("Target ProdTaken is missing or contains null values")
print(f"Dataset registered: {df.shape[0]} rows, {df.shape[1]} columns")
print(df["ProdTaken"].value_counts(dropna=False))
