import numpy as np

def log_marginal_likelihood(Phi, t, alpha, beta):
    """Computes the log of the marginal likelihood."""
    
    Sigma = 1/alpha * Phi @ Phi.T + 1/beta * np.eye(Phi.shape[0])
    
    _, logdet = np.linalg.slogdet(2*np.pi*Sigma)
    
    gauss_exp = np.squeeze(t.T @ np.linalg.inv(Sigma) @ t)
    
    return -0.5 * (logdet + gauss_exp)


def posterior(Phi, t, alpha, beta, return_inverse=False):
    """Computes mean and covariance matrix of the posterior distribution."""
    S_N_inv = alpha * np.eye(Phi.shape[1]) + beta * Phi.T @ Phi
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta * S_N @ Phi.T @ t

    if return_inverse:
        return m_N, S_N, S_N_inv
    else:
        return m_N, S_N

def EM(Phi, t, alpha_0=1e-5, beta_0=1e-5, max_iter=200, rtol=1e-5, verbose=False):
    
    N, M = Phi.shape

    beta = beta_0
    alpha = alpha_0
    
    mlls = []

    for i in range(max_iter):
        beta_prev = beta
        alpha_prev = alpha

        # E-step
        m_N, S_N = posterior(Phi, t, alpha, beta)

        # M-step
        # ADD YOUR CODE HERE
        alpha = M/(np.linalg.norm(m_N,'fro')**2+np.trace(S_N))
        A=t - Phi @ m_N
        beta = N/(np.linalg.norm(A,'fro')**2+np.trace(Phi @ S_N @ Phi.T))
        
        # Compute log-marginal likelihood
        mlls.append(log_marginal_likelihood(Phi, t, alpha=alpha, beta=beta))
        
        if np.isclose(alpha_prev, alpha, rtol=rtol) and np.isclose(beta_prev, beta, rtol=rtol):
            if verbose:
                print(f'Convergence after {i + 1} iterations.')
            return alpha, beta, m_N, S_N, mlls

    if verbose:
        print(f'Stopped after {max_iter} iterations.')

    return alpha, beta, m_N, S_N, mlls