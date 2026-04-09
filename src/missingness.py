import numpy as np
from utils import sigmoid
import pandas as pd

class MissingDataGenerator:
    """
    Generates missing data masks based on MCAR, MAR, and MNAR mechanisms.
    Missing labels are denoted by replacing the true label with -1.
    """
    
    @staticmethod
    def apply_mcar(y, c=0.3, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
            
        y_arr = np.asarray(y)
        y_obs = y_arr.copy()
        
        S = np.random.binomial(1, c, size=len(y_arr))
        y_obs[S == 1] = -1
        
        # Return as Pandas Series if it originally was one
        if hasattr(y, 'index'):
            return pd.Series(y_obs, index=y.index)
        return y_obs

    @staticmethod
    def apply_mar(X, y, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
            
        # Force pure NumPy arrays to prevent Pandas index alignment bugs
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        
        n_samples, n_features = X_arr.shape
        y_obs = y_arr.copy()
        
        w_X = np.random.randn(n_features)
        
        z = np.dot(X_arr, w_X)
        z = (z - np.mean(z)) / (np.std(z) + 1e-8)
        probas = sigmoid(z) 
        
        S = np.random.binomial(1, probas)
        y_obs[S == 1] = -1
        
        if hasattr(y, 'index'):
            return pd.Series(y_obs, index=y.index)
        return y_obs
    
    @staticmethod
    def apply_mnar(X, y, y_weight=2.0, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
            
        # 1. Force strict float types
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        
        # 2. BULLETPROOFING: Replace any stray NaNs in the features with 0.0
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        
        n_samples, n_features = X_arr.shape
        y_obs = y_arr.copy()
        
        w_X = np.random.randn(n_features)
        
        # Now this math is 100% safe from NaNs!
        z = np.dot(X_arr, w_X) + y_weight * y_arr
        z = (z - np.mean(z)) / (np.std(z) + 1e-8)
        probas = sigmoid(z)
        
        S = np.random.binomial(1, probas)
        y_obs[S == 1] = -1
        
        if hasattr(y, 'index'):
            return pd.Series(y_obs, index=y.index)
        return y_obs