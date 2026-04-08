import numpy as np
import logging

logger = logging.getLogger(__name__)

class UnlabeledLogReg:
    """
    Class for training a logistic regression model on a dataset with missing labels.
    
    This class implements semi-supervised learning techniques for unlabeled data 
    (where the target variable is missing, denoted by -1).
    
    Parameters:
    model : FISTA
        An instance of the custom FISTA logistic regression model.
    method : str, default='pseudo_labeling'
        The algorithm name to use for completing missing labels. 
        Options: 'pseudo_labeling' and 'em'.
    """

    def __init__(self, model, method='pseudo_labeling'):
        """
        Initializes the UnlabeledLogReg class.
        """
        self.model = model
        if method not in ['pseudo_labeling', 'em']:
            raise ValueError(f"Invalid method '{method}'. Choose 'pseudo_labeling' or 'em'.")
        self.method = method

    def fit(self, X, y_obs) -> "UnlabeledLogReg":
        """
        Fits the model to the training data using the specified semi-supervised algorithm.
        
        Parameters:
        X : ndarray of shape (n_samples, n_features)
            Training feature vectors for all samples (both labeled and unlabeled).
        y_obs : ndarray of shape (n_samples,)
            Target vector containing observed labels (0 or 1) and missing labels (-1).
            
        Returns:
        self : UnlabeledLogReg
            The fitted class instance.
        """
        if self.method == 'pseudo_labeling':
            return self._pseudo_labeling(X, y_obs)
        elif self.method == 'em':
            return self._em_algorithm(X, y_obs)
        else:
            raise ValueError(f"Unknown method '{self.method}'. Choose 'pseudo_labeling' or 'em'.")

    def _pseudo_labeling(self, X, y_obs) -> "UnlabeledLogReg":
        """
        Implements the Pseudo-labeling algorithm.
        
        Algorithm steps:
        1. Split the dataset into labeled (y != -1) and unlabeled (y == -1) subsets.
        2. Fit the base FISTA model exclusively on the labeled subset.
        3. Predict labels (0 or 1) for the unlabeled subset using the trained base model.
        4. Impute the missing values in the original target vector with the predicted pseudo-labels.
        5. Refit the FISTA model on the completely labeled (augmented) dataset.
        
        Parameters:
        X : ndarray of shape (n_samples, n_features)
            Training feature vectors.
        y_obs : ndarray of shape (n_samples,)
            Target vector with missing labels denoted as -1.
            
        Returns:
        self : UnlabeledLogReg
            The fitted class instance.
        """
        labeled_mask = (y_obs != -1)
        
        X_labeled = X[labeled_mask]
        y_labeled = y_obs[labeled_mask]
        X_unlabeled = X[~labeled_mask]
        
        if len(X_unlabeled) == 0:
            logger.warning("No unlabeled samples found. Performing standard training.")
            self.model.fit(X, y_obs)
            return self

        self.model.fit(X_labeled, y_labeled)
        pseudo_labels = self.model.predict(X_unlabeled)

        y_complete = y_obs.copy()
        y_complete[~labeled_mask] = pseudo_labels

        self.model.is_fitted = False
        self.model.fit(X, y_complete)
        
        return self

    def _em_algorithm(self, X, y_obs, max_em_iter=15, tol=1e-3) -> "UnlabeledLogReg":
        """
        Implements the Expectation-Maximization (EM) algorithm for soft-labeling.
        
        Algorithm steps:
        1. Split the dataset into labeled (y != -1) and unlabeled (y == -1) subsets.
        2. Fit the base FISTA model exclusively on the labeled subset.
        3. Loop until convergence or max_em_iter:
           a. Predict soft probabilities for the unlabeled subset using the current model.
           b. Impute the missing values in the target vector with these probabilities.
           c. Refit the FISTA model on the completely labeled (augmented) dataset using 
              previous weights as a warm start.
        
        Parameters:
        X : ndarray of shape (n_samples, n_features)
            Training feature vectors.
        y_obs : ndarray of shape (n_samples,)
            Target vector with missing labels denoted as -1.
        max_em_iter : int, default=15
            Maximum number of iterations for the EM loop.
        tol : float, default=1e-3
            Tolerance for the convergence criteria based on coefficient changes.
            
        Returns:
        self : UnlabeledLogReg
            The fitted class instance.
        """
        labeled_mask = (y_obs != -1)
        
        X_labeled = X[labeled_mask]
        y_labeled = y_obs[labeled_mask]
        X_unlabeled = X[~labeled_mask]
        
        if len(X_unlabeled) == 0:
            logger.warning("No unlabeled samples found. Performing standard training.")
            self.model.fit(X, y_obs)
            return self

        self.model.fit(X_labeled, y_labeled)
        
        y_complete = y_obs.copy().astype(float)
        
        for em_iter in range(max_em_iter):
            old_betas = self.model.betas.copy()
            
            probas = self.model.predict_proba(X_unlabeled)[:, 1]
            y_complete[~labeled_mask] = probas
            
            self.model.is_fitted = False
            self.model.betas_start = old_betas
            self.model.fit(X, y_complete)
            
            diff = np.linalg.norm(self.model.betas - old_betas)
            if diff < tol:
                logger.info(f"EM converged at iteration {em_iter + 1}.")
                break
                
        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Estimates the class probabilities for the provided samples using the finalized model.
        
        Parameters:
        X : ndarray of shape (n_samples, n_features)
            Feature vectors of the samples to predict.
            
        Returns:
        proba : ndarray of shape (n_samples, 2)
            The predicted probabilities for class 0 and class 1.
        """
        return self.model.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        """
        Predicts class labels for the provided samples using the finalized model.
        
        Parameters:
        X : ndarray of shape (n_samples, n_features)
            Feature vectors of the samples to predict.
            
        Returns:
        preds : ndarray of shape (n_samples,)
            The predicted binary class labels (0 or 1).
        """
        return self.model.predict(X)