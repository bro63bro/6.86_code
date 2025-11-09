"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture

import time
from tqdm import tqdm

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    # First intiialize an empty (n x K) matrix and other variables
    n = X.shape[0]
    d = X.shape[1]
    K = mixture.mu.shape[0]
    softcount_matrix = np.zeros((n, K))
    mask = X != 0
    
    # Create helper functions
    
    # Log of the probability that x^{u} belongs to cluster j
    def N_log(u, j):
        
        diff = X[u][mask[u]] - mixture.mu[j][mask[u]]
        norm_sq = np.dot(diff, diff)
    
        return -(1/2)*np.sum(mask[u]) * np.log(2*np.pi*mixture.var[j]) - (norm_sq / (2 * mixture.var[j]))

    # Dummy function f for ease of calculation
    def f(u, i):
        return np.log(mixture.p[i] + 1e-16) + N_log(u, i)
    
    # This is log(p(j|u)) to operate in the log domain
    def l(j, u):
        variable = logsumexp([f(u, t) for t in range(K)])
        return f(u, j) - variable
    
    # Update softcount matrix entries 
    for i in range(n):
        for j in range(K):
            softcount_matrix[i, j] = np.exp(l(j, i))
    
    # Finally we need to calculate log-likelihood of current assignment
    
    log_likelihood = 0
    for i in range(n):
        variable = logsumexp([f(i, j) for j in range(K)])
        log_likelihood += variable
    return (softcount_matrix, log_likelihood)


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    
    # Instantiate GaussianMixture object and define variables
    gm = GaussianMixture
    n = X.shape[0]
    d = X.shape[1]
    K = post.shape[1]
    mask = X != 0
    
    # Create helper functions
    def mu_hat(k):
        
        # Initialize vector 
        mu_k = np.zeros(d)
        for l in range(d):
            
            # Find mask for observed entries in column l
            # This is all the users who saw feature l
            observed = mask[:, l]
            
            # Find relevant posteriors and values
            
            # For all people who saw feature l, find the probability they belong to cluster k
            weights = post[observed, k]     
            values = X[observed, l]
            
            if np.sum(weights) >= 1:
                mu_k[l] = np.dot(weights, values) / np.sum(weights)
            else:
                mu_k[l] = mixture.mu[k, l]
        return mu_k
        
    """    
    def var_hat(k, mu_k):
        numerator = 0
        denominator = 0
        
        for i in range(n):
            diff = X[i][mask[i]] - mu_k[mask[i]]
            numerator += post[i, k] * np.dot(diff, diff)
            denominator += post[i, k]*np.sum(mask[i])
        
        if denominator > 0 and numerator/denominator > min_variance:
            return numerator/denominator
        else:
            return min_variance
    """
    
    def var_hat(k, mu_k):
        numerator = 0
        denominator = 0
        
        for i in range(n):
            diff = X[i][mask[i]] - mu_k[mask[i]]
            norm_sq = np.dot(diff, diff)
            numerator += post[i, k] * norm_sq
            denominator += post[i, k] * np.sum(mask[i])
        
        if denominator > 0 and numerator/denominator > min_variance:
            return numerator/denominator
        else:
            return min_variance
    """        
    def p_hat(k):
        numerator = 0
        
        for i in range(n):
            numerator += post[i, k]
        return numerator/n
    """
    
    def p_hat(k):
        return np.sum(post[:, k]) / n
    
    # Since tuples are immutable we create new arrays that we can assign stuff to
    # And then convert back to tuples for the GaussianMixture class
    mu_array = np.zeros((K, d))
    var_array = np.zeros((K,))
    p_array = np.zeros((K,))
    
    for j in range(K):
        mu_k = mu_hat(j)      # cache these values once
        mu_array[j] = mu_k    # use it here
        var_array[j] = var_hat(j, mu_k)     # use it here too
        p_array[j] = p_hat(j)
    
    return gm(mu = mu_array, var = var_array, p = p_array)



def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    # While convergence criteria is not met,
    # alternate between E-step and M-step
    old_ll = 69
    new_ll = 420
    iteration = 0
    
    while abs(new_ll - old_ll) > (1e-6*abs(new_ll)):
        
        iteration += 1
        print(f"[EM Iteration {iteration}] Log-likelihood: {new_ll:.16f}")
        
        # Update log-likelihoods
        old_ll = new_ll
        
        # E step
        post, new_ll = estep(X, mixture)
        
        
        # M step
        mixture = mstep(X, post, mixture)
        
    return (mixture, post, new_ll)


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    
    # We want to assign each user's missing values to be
    # the expected value of the mu's over all the clusters
    # That is: we want X[u][l] = sum over j of p(j|u) * mu[j][l] for missing indices l
    
    reverse_mask = X == 0
    
    # First generate the softcount matrix again
    # Intiialize an empty (n x K) matrix and other variables
    n = X.shape[0]
    d = X.shape[1]
    K = mixture.mu.shape[0]
    softcount_matrix = np.zeros((n, K))
    mask = X != 0
    # Create helper functions
    
    
    # Log of the probability that x^{u} belongs to cluster j
    def N_log(u, j):
        
        diff = X[u][mask[u]] - mixture.mu[j][mask[u]]
        norm_sq = np.dot(diff, diff)
    
        return -(1/2)*np.sum(mask[u]) * np.log(2*np.pi*mixture.var[j]) - (norm_sq / (2 * mixture.var[j]))

    # Dummy function f for ease of calculation
    def f(u, i):
        return np.log(mixture.p[i] + 1e-16) + N_log(u, i)
    
    # This is log(p(j|u)) to operate in the log domain
    def l(j, u):
        variable = logsumexp([f(u, t) for t in range(K)])
        return f(u, j) - variable
    
    # Update softcount matrix entries 
    for i in range(n):
        for j in range(K):
            softcount_matrix[i, j] = np.exp(l(j, i))
    
    post = softcount_matrix
    
    
    # Note that for X[u]'s missing l-th component, we have to multiply two vectors:
    # they are the u-th row of post and the l-th column of mu
    # This suggests we just generate a matrix of all values first
    # and then update only the masked values somehow
    
    expected_values = np.matmul(post, mixture.mu)
    X_pred = X.copy()
    
    X_pred[reverse_mask] = expected_values[reverse_mask]

    return X_pred
    
    
    
    
