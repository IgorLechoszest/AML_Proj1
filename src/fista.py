import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    recall_score, 
    precision_score, 
    f1_score, 
    balanced_accuracy_score, 
    roc_auc_score, 
    average_precision_score
)
import logging
logger = logging.getLogger(__name__)

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
class FISTA:
    """
    Logistic Regression classifier with L1 penalty (Lasso) optimized using FISTA.
    
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) is used to minimize 
    the Log-Loss function combined with an L1 regularization term, allowing for 
    feature selection.
    
    Parameters:
    lambdas : array-like, default=None
        List of lambda values for the regularization path. If None, a default range (25 values between 0.0001 and 10) will be used.
    lr : float, default=0.01
        Learning rate (step size) for the gradient descent step.
    max_iter : int, default=1000
        Maximum number of iterations for the optimization solver.
    tol : float, default=1e-4
        Tolerance for the stopping criteria for early stopping.
    betas_start: ndarray of shape (n_features,)
        Initial weights vector. If None, it will be initialized to zeros.
    start_point: ndarray of shape (n_features,)
        Initial point for the optimization. If None, it will be initialized to zeros.
    start_momentum: float, default=1.0
        Starting value for the momentum. Typically set to 1.0.
        
    Attributes:
    coef_path : list of ndarray
        List of coefficient vectors for each lambda in the regularization path.
    val_scores : list of float
        List of validation scores corresponding to each lambda in the regularization path.
    best_lambda : float
        The lambda value that achieved the best validation score.        
    betas : ndarray of shape (n_features,)
        Vector of weights learned by the model after fitting.
    classes : ndarray of shape (n_classes,)
        A list of class labels.
    is_fitted : bool
        Indicates whether the model has been fitted.    
    """

    def __init__(self, lambdas=None, lr=0.01, max_iter=1000, tol=1e-4, betas_start=None, start_point=None, start_momentum=1.0):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.betas_start = betas_start
        self.start_point = start_point
        self.start_momentum = start_momentum
        if lambdas is None:
            self.lambdas = np.logspace(-4, 1, 25)
        else:
            self.lambdas = np.array(lambdas)

        self.coef_path = []
        self.val_scores = []
        self.best_lambda = None

        self.betas = None
        self.classes = None
        self.is_fitted = False
    def _soft_thresholding(self, betas, threshold) -> np.ndarray:
        """
        Soft-thresholding operator for L1 regularization. Allows for L1 penalty to set coefficients to zero, enabling feature selection.

        Parameters:
        betas : ndarray
            Current coefficient vector after gradient step.
        threshold : float
            Threshold value (learning_rate * lambda_val).
            
        Returns:
        Updated coefficient vector after applying soft-thresholding.
        """
        return np.sign(betas) * np.maximum(0, np.abs(betas) - threshold)

    def _compute_gradient(self, X, y, beta_current) -> np.ndarray:
        """
        Computes the gradient of the Log-Loss function.
        
        Parameters:
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values 0 or 1.
        beta_current : ndarray of shape (n_features,)
            Current weight vector.
        Returns:
        Gradient vector of shape (n_features,).
        """
        n = X.shape[0]
        predictions = sigmoid(X @ beta_current)
        grad = (1 / n) * X.T @ (predictions - y)
        return grad

    def fit(self, X, y) -> "FISTA":
        """
        Fit the model to the given training data using FISTA.
        Algorithm steps:
        1. Initialize weights and momentum (if not provided).
        2. For each iteration:
           a. Compute the gradient of the Log-Loss at the current point: gradient = 1/n * X.T @ (sigmoid(X @ beta_current) - y).
           b. Take a gradient step to get an intermediate weight vector: beta_intermediate = beta_current - self.lr * gradient.
           c. Apply the soft-thresholding operator to add L1 regularization: beta_next = self._soft_thresholding(beta_intermediate, self.lr * self.lambda_val).
           d. Update the momentum and compute the next point for the gradient step: mu_next = (1+sqrt(1+4*mu**2))/2, y_next = beta_next + (mu - 1) / mu_next * (beta_next - beta_current).
           e. Check for convergence based on the change in weights or maximum iterations.
        
        Parameters:
        X : ndarray of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : ndarray of shape (n_samples,)
            Target vector relative to X. Must be binary (0 or 1).
        Returns:
        Fitted estimator.
        """
        if self.is_fitted:
            logger.warning("Model is already fitted. Re-fitting will overwrite previous results.")

        n_features = X.shape[1]
        self.classes = np.unique(y) 
        self.betas = np.zeros(n_features) if self.betas_start is None else self.betas_start.copy()  
        self.coef_path = []

        for lmbda in self.lambdas:
            beta_current = self.betas.copy()
            y_current = beta_current.copy() if self.start_point is None else self.start_point.copy()
            mu = self.start_momentum
            for iter in range(self.max_iter):
                gradient = self._compute_gradient(X, y, y_current)
                beta_intermediate = y_current - self.lr * gradient
                beta_next = self._soft_thresholding(beta_intermediate, self.lr * lmbda)

                mu_next = (1 + np.sqrt(1 + 4 * mu ** 2)) / 2
                y_next = beta_next + (mu - 1) / mu_next * (beta_next - beta_current)

                if np.linalg.norm(beta_next - beta_current) < self.tol:
                    logger.info(f"Convergence reached at iteration {iter} for lambda={lmbda}. Stopping early.")
                    break

                beta_current = beta_next
                y_current = y_next
                mu = mu_next

            self.coef_path.append(beta_current.copy())
            
        self.coef_path = np.array(self.coef_path)
        self.betas = self.coef_path[0]  # Default to the first lambda's coefficients, will be updated after validation 
        self.is_fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Estimate the probability of the sample for each class in the model.
        Algorithm steps:
        1. Compute the linear combination of inputs and weights: logits = X @ self.betas.
        2. Apply the sigmoid function to get probabilities.
        
        Parameters:
        X : ndarray of shape (n_samples, n_features)
            Vector of samples.
            
        Returns:
        proba : ndarray of shape (n_samples, 2)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in self.classes.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() before predict_proba().")
        probas = sigmoid(X @ self.betas)
        return np.column_stack((1 - probas, probas))

    def predict(self, X) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Parameters:
        X : ndarray of shape (n_samples, n_features)
            Samples vector.
            
        Returns:
        C : ndarray of shape (n_samples,)
            Vector of predicted class labels per sample (0 or 1).
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() before predict().")
        probas = self.predict_proba(X)[:, 1]
        return (probas >= 0.5).astype(int)
    def validate(self, X_valid, y_valid, measure='f1') -> float:
        """
        Evaluates the model on validation data across all lambdas and selects the best one.
        """
        self.val_scores = []
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() before validate().")
        
        for beta in self.coef_path:
            self.betas = beta
            y_proba = self.predict_proba(X_valid)[:, 1]
            y_pred = self.predict(X_valid)
            
            if measure == 'recall':
                score = recall_score(y_valid, y_pred)
            elif measure == 'precision':
                score = precision_score(y_valid, y_pred)
            elif measure == 'f1':
                score = f1_score(y_valid, y_pred)
            elif measure == 'balanced_accuracy':
                score = balanced_accuracy_score(y_valid, y_pred)
            elif measure == 'roc_auc':
                score = roc_auc_score(y_valid, y_proba)
            elif measure == 'pr_auc':
                score = average_precision_score(y_valid, y_proba)
            else:
                raise ValueError(f"Unknown measure: {measure}")
                
            self.val_scores.append(score)

        best_idx = np.argmax(self.val_scores)
        self.best_lambda = self.lambdas[best_idx]
        self.betas = self.coef_path[best_idx]
        return self.best_lambda

    def plot(self, measure) -> None:
        """Plots the evaluation measure against lambda values."""
        if not self.val_scores:
            raise ValueError("Model is not fitted. Call fit() and validate() before plot().")
        plt.figure(figsize=(8, 5))
        plt.plot(self.lambdas, self.val_scores, marker='o')
        plt.xscale('log')
        plt.xlabel('Lambda')
        plt.ylabel(measure)
        plt.title(f'{measure} vs Lambda')
        plt.grid(True)
        plt.show()

    def plot_coefficients(self) -> None:
        """Plots the coefficient paths as a function of lambda."""
        if len(self.coef_path) == 0:
            raise ValueError("Model is not fitted. Call fit() before plot_coefficients().")
        plt.figure(figsize=(10, 6))
        for i in range(self.coef_path.shape[1]):
            plt.plot(self.lambdas, self.coef_path[:, i])
        plt.xscale('log')
        plt.xlabel('Lambda')
        plt.ylabel('Coefficient values')
        plt.title('Coefficient Paths (with L1 Regularization)')
        plt.grid(True)
        plt.show()