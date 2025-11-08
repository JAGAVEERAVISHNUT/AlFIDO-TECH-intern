import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ], remainder='drop', sparse_threshold=0)

    return preprocessor


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate a model using the project preprocessing')
    parser.add_argument('--csv', default='dataset1.csv', help='Path to CSV file (relative to script)')
    parser.add_argument('--target', default='target', help='Name of the target column')
    parser.add_argument('--model', choices=['logreg', 'tree'], default='logreg', help='Model to train')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--save-model', action='store_true', help='Save trained model to model.joblib')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(base_dir, args.csv)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please provide a valid CSV path.")

    df = pd.read_csv(csv_path)
    # Drop common identifier columns if present
    for c in ['id', 'name']:
        if c in df.columns:
            df = df.drop(c, axis=1)

    if args.target not in df.columns:
        raise KeyError(f"`{args.target}` not found in the CSV. Please ensure the target column exists.")

    X = df.drop(args.target, axis=1)
    y = df[args.target]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Split first (avoid leakage). Stratify when target looks categorical/small-cardinality
    stratify = y if (hasattr(y, 'nunique') and y.nunique() <= 20) else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=stratify)

    # Try to reuse an existing preprocessor if available
    preprocessor_path = os.path.join(base_dir, 'preprocessor.joblib')
    if os.path.exists(preprocessor_path):
        try:
            preprocessor = joblib.load(preprocessor_path)
            print(f"Loaded preprocessor from {preprocessor_path}")
            # best-effort: assume it's already fitted; if not, fit below
            try:
                # check if transformer has been fitted by trying transform on a small sample
                _ = preprocessor.transform(X_train.head(1))
                fitted = True
            except Exception:
                fitted = False
        except Exception:
            print("Failed to load existing preprocessor; will build a new one.")
            preprocessor = None
            fitted = False
    else:
        preprocessor = None
        fitted = False

    if preprocessor is None:
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    if not fitted:
        preprocessor.fit(X_train)

    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Choose model
    if args.model == 'logreg':
        model = LogisticRegression(max_iter=2000)
    else:
        model = DecisionTreeClassifier(random_state=42)

    # Train
    model.fit(X_train_trans, y_train)

    # Predict & evaluate
    y_pred = model.predict(X_test_trans)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\nModel training complete")
    print(f"Model: {args.model}")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:\n", report)
    print("Confusion matrix:\n", cm)

    # If binary and model supports predict_proba, print ROC AUC
    try:
        if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_test_trans)[:, 1]
            auc = roc_auc_score(y_test, probs)
            print(f"ROC AUC: {auc:.4f}")
    except Exception:
        pass

    if args.save_model:
        out_path = os.path.join(base_dir, 'model.joblib')
        joblib.dump({'preprocessor': preprocessor, 'model': model}, out_path)
        print(f"Saved preprocessor+model bundle to {out_path}")


if __name__ == '__main__':
    main()
