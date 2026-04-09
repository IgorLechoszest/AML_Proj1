
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function for arrays.
    exp(-z) can overflow for z<0 and exp(z) can overflow for z>0 so we handle positive and negative cases separately.
    """
    res = np.zeros_like(z, dtype=float)

    pos_mask = z >= 0
    neg_mask = ~pos_mask

    res[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))

    exp_z_neg = np.exp(z[neg_mask])
    res[neg_mask] = exp_z_neg / (1.0 + exp_z_neg)
    
    return res