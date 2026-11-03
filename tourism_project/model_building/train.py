import os
from pathlib import Path
import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
Xtrain, Xtest = pd.read_csv("Xtrain.csv"), pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()
numeric = Xtrain.select_dtypes(include=["number"]).columns.tolist()
categorical = Xtrain.select_dtypes(exclude=["number"]).columns.tolist()
weight = float((ytrain == 0).sum() / max((ytrain == 1).sum(), 1))
pre = make_column_transformer((StandardScaler(), numeric), (OneHotEncoder(handle_unknown="ignore"), categorical))
model = xgb.XGBClassifier(scale_pos_weight=weight, random_state=42, eval_metric="logloss", n_jobs=2)
pipeline = make_pipeline(pre, model)
grid = {"xgbclassifier__n_estimators":[100,200], "xgbclassifier__max_depth":[3,5], "xgbclassifier__learning_rate":[0.05,0.1], "xgbclassifier__reg_lambda":[1,5]}
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("Tourism Package Prediction")
with mlflow.start_run():
    search = GridSearchCV(pipeline, grid, cv=3, scoring="f1", n_jobs=-1)
    search.fit(Xtrain, ytrain)
    best = search.best_estimator_
    threshold = 0.45
    probabilities = best.predict_proba(Xtest)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    report = classification_report(ytest, predictions, output_dict=True, zero_division=0)
    mlflow.log_params(search.best_params_)
    mlflow.log_metrics({"accuracy":report["accuracy"], "precision":report["1"]["precision"], "recall":report["1"]["recall"], "f1":report["1"]["f1-score"], "roc_auc":roc_auc_score(ytest, probabilities)})
    out = ROOT / "deployment" / "best_tourism_package_model_v1.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, out)
    mlflow.log_artifact(str(out), artifact_path="model")
    print("Best parameters:", search.best_params_)
    print("Test metrics:", {k:report[k] for k in ["accuracy"]}, report["1"])
    print("Saved model:", out)
