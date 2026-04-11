import numpy as np
from utils import sigmoid
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class MissingDataGenerator:
    """
    Generates missing data masks based on MCAR, MAR1, MAR2, and MNAR mechanisms.
    Missing labels are denoted by replacing the true label with -1.
    Strictly returns (X, y_obs) to fulfill project requirements.
    """
    
    @staticmethod
    def apply_mcar(X, y, c=0.3, random_state=None):
        """MCAR: Missingness depends on a flat probability 'c'."""
        if random_state is not None:
            np.random.seed(random_state)
            
        y_arr = np.asarray(y)
        y_obs = y_arr.copy()
        
        S = np.random.binomial(1, c, size=len(y_arr))
        y_obs[S == 1] = -1
        
        # Return tuple (X, y_obs)
        if hasattr(y, 'index'):
            return X, pd.Series(y_obs, index=y.index)
        return X, y_obs

    @staticmethod
    def apply_mar1(X, y, feature_idx=None, random_state=None):
        """MAR1: Missingness depends ONLY on a single explanatory variable."""
        if random_state is not None:
            np.random.seed(random_state)
            
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        
        n_samples, n_features = X_arr.shape
        y_obs = y_arr.copy()
        
        if feature_idx is None:
            feature_idx = np.random.randint(0, n_features)
            
        w_x = np.random.randn()
        z = X_arr[:, feature_idx] * w_x
        z = (z - np.mean(z)) / (np.std(z) + 1e-8)
        probas = sigmoid(z) 
        
        S = np.random.binomial(1, probas)
        y_obs[S == 1] = -1
        
        # Return tuple (X, y_obs)
        if hasattr(y, 'index'):
            return X, pd.Series(y_obs, index=y.index)
        return X, y_obs

    @staticmethod
    def apply_mar2(X, y, random_state=None):
        """MAR2: Missingness depends on ALL explanatory variables."""
        if random_state is not None:
            np.random.seed(random_state)
            
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        
        n_samples, n_features = X_arr.shape
        y_obs = y_arr.copy()
        
        w_X = np.random.randn(n_features)
        z = np.dot(X_arr, w_X)
        z = (z - np.mean(z)) / (np.std(z) + 1e-8)
        probas = sigmoid(z) 
        
        S = np.random.binomial(1, probas)
        y_obs[S == 1] = -1
        
        # Return tuple (X, y_obs)
        if hasattr(y, 'index'):
            return X, pd.Series(y_obs, index=y.index)
        return X, y_obs
    
    @staticmethod
    def apply_mnar(X, y, y_weight=2.0, random_state=None):
        """MNAR: Missingness depends on BOTH X and Y."""
        if random_state is not None:
            np.random.seed(random_state)
            
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        
        n_samples, n_features = X_arr.shape
        y_obs = y_arr.copy()
        
        w_X = np.random.randn(n_features)
        z = np.dot(X_arr, w_X) + y_weight * y_arr
        z = (z - np.mean(z)) / (np.std(z) + 1e-8)
        probas = sigmoid(z)
        
        S = np.random.binomial(1, probas)
        y_obs[S == 1] = -1
        
        # Return tuple (X, y_obs)
        if hasattr(y, 'index'):
            return X, pd.Series(y_obs, index=y.index)
        return X, y_obs