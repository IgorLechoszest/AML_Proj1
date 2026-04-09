import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.preprocessing import LabelEncoder

def create_artificial_dataset(n_samples=2000, random_state=2137):
    """
    Creates an artificial dataset for classification tasks.
    Parameters:
    - n_samples: Number of samples to generate.
    - random_state: Seed for reproducibility.
    Returns:
    - df_X: DataFrame containing the features.
    - df_y: Series containing the target variable.
    """

    X, y = make_classification(
        n_samples=n_samples,
        n_features=50,          # Large number of features for complexity
        n_informative=20,       # Number of features that actually matter
        n_redundant=15,         # Injects collinearity on purpose!
        n_classes=2,            # Binary classification
        random_state=random_state
    )
    
    feature_names = [f"Feature_{i}" for i in range(50)]
    df_X = pd.DataFrame(X, columns=feature_names)
    df_y = pd.Series(y, name="Target")
    
    return df_X, df_y

def load_spambase():
    """
    Loads the Spambase dataset (Binary Classification).
    Target: 1 (Spam), 0 (Non-Spam).
    Features: 57 numerical features.
    """
    print("Loading Spambase dataset...")
    # Fetch from OpenML (ID 44 is Spambase)
    data = fetch_openml(name='spambase', version=1, as_frame=True, parser='auto')
    X = data.data
    y = data.target
    
    # Ensure target is strictly integer 0 and 1
    y = y.astype(int)
    
    # Basic cleanup just in case
    X = X.apply(pd.to_numeric, errors='coerce').fillna(X.mean())
    
    print(f"Spambase loaded: X shape {X.shape}, y shape {y.shape}")
    return X, y

def load_sonar():
    """
    Loads the Sonar dataset (Binary Classification).
    Robustly encodes string targets to integers.
    Features: 60 numerical features.
    """
    print("Loading Sonar dataset...")
    # Fetch from OpenML
    data = fetch_openml(name='sonar', version=1, as_frame=True, parser='auto')
    X = data.data
    y = data.target
    
    # 1. Drop any rows where the target itself is genuinely missing
    valid_indices = y.dropna().index
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]
    
    # 2. Robust target encoding: Handles any string labels safely
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), index=X.index)
    
    # 3. Basic cleanup for the features
    X = X.apply(pd.to_numeric, errors='coerce').fillna(X.mean())
    
    print(f"Sonar loaded: X shape {X.shape}, y shape {y.shape}")
    return X, y

def load_breast_cancer_data():
    """
    Loads the Breast Cancer Wisconsin dataset (Binary Classification).
    Target: 0 (Malignant) and 1 (Benign).
    Features: 30 continuous numerical features.
    """
    print("Loading Breast Cancer dataset...")
    
    # Fetch directly from scikit-learn's local, guaranteed datasets
    data = load_breast_cancer(as_frame=True)
    
    X = data.data
    y = data.target 
    
    print(f"Breast Cancer loaded: X shape {X.shape}, y shape {y.shape}")
    print(f"Positive labels (1s): {y.sum()} out of {len(y)}")
    
    return X, y