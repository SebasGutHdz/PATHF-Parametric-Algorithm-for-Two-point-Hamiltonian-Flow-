

from flax import nnx
import flax
import jax
import jax.numpy as jnp 
from jaxtyping import Array



def guassian_wasserstein_squared_distance(theta: Array, sigma: Array)-> float:
    '''
    Compute the squared 2-Wasserstein distance between two Gaussian distributions. 
    Recall
    W_2^2(N(0, C1), N(0, C2)) = Tr(C1 + C2 - 2*((C2)^(1/2) * (C1) * (C2)^(1/2))^(1/2))^(1/2)
    Args:
        theta: Parameteric representation, the covariance matrix is given by theta^T @ theta.
        sigma: Covariance matrix of the second Gaussian distribution.
    
    Returns:
        The squared 2-Wasserstein distance.    
    '''
    # Verify that dimensions match
    d = theta.shape[1]
    assert sigma.shape == (d,d), "Dimension mismatch between theta and sigma."
    # Compute covariance matrix
    cov = theta.T@theta
    # Compute the square root of sigma. To ensure differentiability, use linalg.eigh
    eigvals_target, eigvecs_target = jnp.linalg.eigh(sigma)
    sqrt_eigvals = jnp.sqrt(eigvals_target)
    sqrt_sigma = eigvecs_target@jnp.diag(sqrt_eigvals)@eigvecs_target.T
    # Compute product term
    prod_term = sqrt_sigma@cov@sqrt_sigma
    # Compute the square root of the product term
    eigvals_prod, eigvecs_prod = jnp.linalg.eigh(prod_term)
    sqrt_eigvals_prod = jnp.sqrt(eigvals_prod)
    sqrt_prod_term = eigvecs_prod @ jnp.diag(sqrt_eigvals_prod) @ eigvecs_prod.T

    sum_matrix = cov+sigma-2*sqrt_prod_term
    trace_term = jnp.trace(sum_matrix)

    return trace_term

