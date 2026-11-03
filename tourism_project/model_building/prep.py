import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "tourism.csv")
df = df.drop(columns=["CustomerID", "Unnamed: 0"], errors="ignore")
target = "ProdTaken"
X, y = df.drop(columns=[target]), df[target]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
for name, value in [("Xtrain", Xtrain), ("Xtest", Xtest), ("ytrain", ytrain), ("ytest", ytest)]:
    value.to_csv(f"{name}.csv", index=False)
print("Train/test split saved.")
