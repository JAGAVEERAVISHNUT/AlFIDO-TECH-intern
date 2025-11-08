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


def build_preprocessor(numeric_cols, categorical_cols):
    """Create a ColumnTransformer that imputes and scales numeric features and imputes+one-hot encodes categoricals."""
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
    parser = argparse.ArgumentParser(description='Preprocess dataset for ML')
    parser.add_argument('--csv', default='dataset1.csv', help='Path to CSV file (relative to script)')
    parser.add_argument('--target', default='target', help='Name of the target column')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--save-pipeline', action='store_true', help='Save fitted preprocessor to preprocessor.joblib')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(base_dir, args.csv)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please provide a valid CSV path.")

    df = pd.read_csv(csv_path)
    print("Original DataFrame (head):")
    print(df.head())

    # If identifier columns exist, drop them safely
    for c in ['id', 'name']:
        if c in df.columns:
            df = df.drop(c, axis=1)

    if args.target not in df.columns:
        raise KeyError(f"`{args.target}` not found in the CSV. Please ensure the target column exists.")

    X = df.drop(args.target, axis=1)
    y = df[args.target]

    # Determine numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")

    # Decide whether to stratify: use stratify if target appears categorical
    stratify = None
    try:
        if y.nunique() <= 20:
            stratify = y
    except Exception:
        stratify = None

    # Split first to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=stratify)

    # Build and fit preprocessor on training data only
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(X_train)

    # Transform train and test
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Try to get feature names for transformed data
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_train_trans.shape[1])]

    # Convert transformed arrays back to DataFrames for readability
    X_train_df = pd.DataFrame(X_train_trans, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test.index)

    print("\nPreprocessing complete.")
    print(f"X_train shape: {X_train_df.shape}")
    print(f"X_test shape: {X_test_df.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    if args.save_pipeline:
        out_path = os.path.join(base_dir, 'preprocessor.joblib')
        joblib.dump(preprocessor, out_path)
        print(f"Saved fitted preprocessor to {out_path}")


if __name__ == '__main__':
    main()
