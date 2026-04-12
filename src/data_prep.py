import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, make_classification
from sklearn.preprocessing import LabelEncoder

def clean_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns to numeric and impute missing values with column means."""
    X_numeric = X.apply(pd.to_numeric, errors='coerce')
    return X_numeric.fillna(X_numeric.mean())


def remove_collinear_features(X: pd.DataFrame, threshold: float = 0.95):
    """Remove highly collinear features based on absolute Pearson correlation."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if (upper[col] > threshold).any()]
    return X.drop(columns=to_drop), to_drop


def prepare_features(X: pd.DataFrame, corr_threshold: float = 0.95):
    """Apply numeric conversion, mean imputation, and collinearity filtering."""
    X_clean = clean_numeric_features(X)
    X_prepared, dropped_cols = remove_collinear_features(X_clean, threshold=corr_threshold)
    return X_prepared, dropped_cols


def _encode_binary_target(y: pd.Series) -> pd.Series:
    """Encode binary labels to 0/1 while preserving the original index."""
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(int)

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y)
    return pd.Series(encoded, index=y.index, name=y.name)


def create_artificial_dataset(n_samples=2000, random_state=2137, corr_threshold=0.95):
    """
    Create an artificial binary classification dataset.

    Parameters:
    - n_samples: Number of observations.
    - random_state: Seed for reproducibility.
    - corr_threshold: Correlation threshold for collinearity filtering.

    Returns:
    - df_X: DataFrame with prepared features.
    - df_y: Series with binary target.
    """

    X, y = make_classification(
        n_samples=n_samples,
        n_features=50,
        n_informative=20,
        n_redundant=15,
        n_classes=2,
        random_state=random_state
    )

    feature_names = [f"Feature_{i}" for i in range(50)]
    df_X = pd.DataFrame(X, columns=feature_names)
    df_X, dropped_cols = prepare_features(df_X, corr_threshold=corr_threshold)
    df_y = pd.Series(y, name="Target")

    print(
        f"Artificial dataset prepared: X shape {df_X.shape}, y shape {df_y.shape}, "
        f"dropped collinear features: {len(dropped_cols)}"
    )
    return df_X, df_y


def load_spambase(corr_threshold=0.95):
    """
    Load the Spambase dataset (binary classification).

    Target: 1 (spam), 0 (not spam).
    """
    print("Loading Spambase dataset...")
    data = fetch_openml(name='spambase', version=1, as_frame=True, parser='auto')
    X = data.data.copy()
    y = _encode_binary_target(data.target)
    X, dropped_cols = prepare_features(X, corr_threshold=corr_threshold)

    print(
        f"Spambase loaded: X shape {X.shape}, y shape {y.shape}, "
        f"dropped collinear features: {len(dropped_cols)}"
    )
    return X, y


def load_sonar(corr_threshold=0.95):
    """
    Load the Sonar dataset (binary classification).

    Target labels are encoded to 0/1.
    """
    print("Loading Sonar dataset...")
    data = fetch_openml(name='sonar', version=1, as_frame=True, parser='auto')
    X = data.data.copy()
    y = data.target.copy()

    valid_indices = y.dropna().index
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]

    y = _encode_binary_target(y)
    X, dropped_cols = prepare_features(X, corr_threshold=corr_threshold)

    print(
        f"Sonar loaded: X shape {X.shape}, y shape {y.shape}, "
        f"dropped collinear features: {len(dropped_cols)}"
    )
    return X, y


def load_breast_cancer_data(corr_threshold=0.95):
    """
    Load the Breast Cancer Wisconsin dataset (binary classification).

    Target: 0 (malignant), 1 (benign).
    """
    print("Loading Breast Cancer dataset...")

    data = load_breast_cancer(as_frame=True)

    X = data.data.copy()
    y = data.target.copy().astype(int)
    X, dropped_cols = prepare_features(X, corr_threshold=corr_threshold)

    print(
        f"Breast Cancer loaded: X shape {X.shape}, y shape {y.shape}, "
        f"dropped collinear features: {len(dropped_cols)}"
    )
    print(f"Positive labels (1s): {int(y.sum())} out of {len(y)}")

    return X, y


def load_ionosphere(corr_threshold=0.95):
    """
    Load the Ionosphere dataset (binary classification) from OpenML.

    Target labels are encoded to 0/1.
    """
    print("Loading Ionosphere dataset...")
    data = fetch_openml(name='ionosphere', version=1, as_frame=True, parser='auto')

    X = data.data.copy()
    y = _encode_binary_target(data.target)
    X, dropped_cols = prepare_features(X, corr_threshold=corr_threshold)

    print(
        f"Ionosphere loaded: X shape {X.shape}, y shape {y.shape}, "
        f"dropped collinear features: {len(dropped_cols)}"
    )
    return X, y
