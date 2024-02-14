import numpy as np
from scipy.stats import norm, truncnorm
from scipy.linalg import cho_solve


H_UPPER= 10

def truncnorm_mean(mu, sigma, left=0, right=1): 
    res = mu - (sigma * (norm.pdf((right-mu) / sigma) 
                         - norm.pdf((left-mu) / sigma)) / 
                        (norm.cdf((right- mu) / sigma) 
                         - norm.cdf((left-mu) / sigma)))
    return  res

def truncnorm_var(mu, sigma, left=0, right=1):
    alpha, beta = (left - mu) / sigma,  (right - mu) / sigma
    res = sigma**2 * (1 - (beta*norm.pdf(beta) - alpha*norm.pdf(alpha)) /
                          (norm.cdf(beta) - norm.cdf(alpha))
                         -((norm.pdf(beta) - norm.pdf(alpha)) /
                          (norm.cdf(beta) - norm.cdf(alpha)))**2)
    return  res
    
def variance_h(mu, sigma, c):
    # compute variance of S given random h
    E = truncnorm_mean(mu, sigma, 0, H_UPPER)
    V = truncnorm_var(mu, sigma, 0, H_UPPER)
    return (norm.cdf(mu / sigma) * (c**2 * V + (c * E)**2) 
            - norm.cdf(mu / sigma)**2 * (c * E)**2)


class AcqTemporal():
    
    def __init__(self, inputs):
        self.inputs = inputs
        
    def compute_value_(self, x):
        '''Integration of std(S) as acq'''
        # get the reduced std
        x = np.atleast_2d(x)
        reduced_std = self.compute_reduced_std(x)
        std_trunc_norm = variance_h(self.mean, reduced_std, self.samples[:,0])
        std_trunc_norm = np.sqrt(std_trunc_norm)
        
        return np.sum(std_trunc_norm) / self.mean.shape[0]
    
    def compute_value(self, x):
        '''Eq. 11 useing indicator function'''
        # get the reduced std
        x = np.atleast_2d(x)
        reduced_std = self.compute_reduced_std(x)
        upper = self.mean + reduced_std
        lower = self.mean - reduced_std
        upper = np.where(upper < 0, 0, upper)
        lower = np.where(lower < 0, 0, lower)
        
        return (abs(np.sum((upper - lower) * self.samples[:,0])) 
                    / self.mean.shape[0])
        
    def update_prior_search(self, model, num_samples=20000):
        rng = np.random.default_rng(0)
        self.model = model
        self.samples = rng.choice(self.inputs.samples, num_samples, False)
        
        self.mean, self.std = self.model.predict(self.samples, 
                                                 return_std=True)
        K_trans_mc = self.model.kernel_(self.samples, self.model.X_train_)
        self.alpha = cho_solve((self.model.L_, True), K_trans_mc.T, 
                               check_finite=False)

    def compute_reduced_std(self, x):
        '''based on iterative Gaussian process Eq.19-20'''
        K_trans_x = self.model.kernel_(x, self.model.X_train_)
        cov_x_mc = self.model.kernel_(x, self.samples) - K_trans_x @ self.alpha
        var_reduction = cov_x_mc ** 2 / ((self.model.predict(x, 
                                            return_std=True)[1])**2)
        reduced_std = np.sqrt(self.std**2 - var_reduction)
        
        return reduced_std