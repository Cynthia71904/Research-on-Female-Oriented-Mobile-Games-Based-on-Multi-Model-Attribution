import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import os
import warnings
warnings.filterwarnings("ignore")

INPUT_CSV = "data2_clean.csv"
TARGET_COLS = [
    "Downloads First 30 Days (WW)",
    "Downloads (Absolute)",
    "Retention_1d",
    "Retention_7d",
    "Retention_14d",
    "Retention_30d",
    "Retention_60d"
]
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_BINS = 3

def safe_qcut(series, q=N_BINS):
    s = series.dropna()
    if s.empty:
        raise ValueError("empty target series")
    try:
        return pd.qcut(series, q=q, labels=False, duplicates='drop').astype(int)
    except Exception:
        ranks = series.rank(method='first')
        return pd.qcut(ranks, q=q, labels=False, duplicates='drop').astype(int)

def prepare_features(df, drop_cols):
    df = df.copy().drop(columns=drop_cols, errors='ignore')
    categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    if categorical_cols:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[categorical_cols] = oe.fit_transform(df[categorical_cols])
    return df

def plot_confusion_matrix(cm, model_name, target_name, save_dir="confusion_matrices"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - {target_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_{target_name}_cm.png")
    plt.close()

def main():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    existing_targets = [c for c in TARGET_COLS if c in df.columns]
    if not existing_targets:
        raise RuntimeError("No target columns found in CSV.")

    drop_cols = ["Name","ID","Date"] + existing_targets
    X_all = prepare_features(df, drop_cols)

    all_metrics = []
    metric_names = ["accuracy", "precision", "recall"]

    for target in existing_targets:
        subset = df[[target]].join(X_all).dropna(subset=[target])
        if subset.empty:
            print(f"  No data for target {target}, skipped.")
            continue

        y = safe_qcut(subset[target], q=N_BINS)
        X = subset.drop(columns=[target])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        for col in X_train.columns:
            if X_train[col].dtype in ['float64', 'int64']:
                X_train[col].fillna(X_train[col].mean(), inplace=True)
                X_test[col].fillna(X_train[col].mean(), inplace=True)
            else:
                X_train[col].fillna(X_train[col].mode()[0], inplace=True)
                X_test[col].fillna(X_train[col].mode()[0], inplace=True)

        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10, random_state=RANDOM_STATE),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, learning_rate=0.1, subsample=1.0, random_state=RANDOM_STATE),
            "LogisticRegression": LogisticRegression(max_iter=1000, tol=0.001, random_state=RANDOM_STATE),
            "NaiveBayes": GaussianNB()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            plot_confusion_matrix(cm, name, target)

            all_metrics.append({
                "model": name,
                "target": target,
                "accuracy": acc,
                "precision": prec,
                "recall": rec
            })

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv("model_metrics.csv", index=False)

    for metric in metric_names:
        plt.figure(figsize=(10,6))
        for model_name in metrics_df['model'].unique():
            subset = metrics_df[metrics_df['model']==model_name]
            plt.plot(subset['target'], subset[metric], marker='o', label=model_name)
        plt.title(f"{metric} across targets")
        plt.xlabel("Target Variable")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{metric}_lineplot.png")
        plt.close()

if __name__ == "__main__":
    main()
