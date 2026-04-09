import numpy as np

def train_naive_model(model, X, y_obs) -> "FISTA":
    """
    Trains the FISTA model using only the observed (labeled) data.
    Ignores all samples where the label is missing (y_obs == -1).
    
    Parameters:
    model : FISTA
        An instance of the custom FISTA logistic regression model.
    X : ndarray of shape (n_samples, n_features)
        Training feature vectors for all samples.
    y_obs : ndarray of shape (n_samples,)
        Target vector containing observed labels (0 or 1) and missing labels (-1).
        
    Returns:
    model : FISTA
        The fitted model trained only on the labeled subset.
    """
    labeled_mask = (y_obs != -1)
    X_labeled = X[labeled_mask]
    y_labeled = y_obs[labeled_mask]
    
    model.is_fitted = False
    model.fit(X_labeled, y_labeled)
    
    return model

def train_oracle_model(model, X, y_true) -> "FISTA":
    """
    Trains the FISTA model using the ground-truth complete data (Oracle).
    This represents the best possible scenario (upper bound of performance).
    
    Parameters:
    model : FISTA
        An instance of the custom FISTA logistic regression model.
    X : ndarray of shape (n_samples, n_features)
        Training feature vectors for all samples.
    y_true : ndarray of shape (n_samples,)
        Ground-truth target vector without any missing labels.
        
    Returns:
    model : FISTA
        The fitted model trained on the complete dataset.
    """
    model.is_fitted = False
    model.fit(X, y_true)
    
    return model