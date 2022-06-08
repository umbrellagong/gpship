import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky, cho_solve, solve_triangular


LOGISTIC_COEFF = 20 

class AcqTemporal():
    
    def __init__(self, inputs):
        self.inputs = inputs
    
    
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
    
    def compute_value_(self, x):  
        '''Eq. 11 useing logistic function'''
        # get the reduced std
        x = np.atleast_2d(x)
        reduced_std = self.compute_reduced_std(x)
        upper = self.mean + reduced_std
        lower = self.mean - reduced_std
        upper = (1 / (1 + np.exp(-LOGISTIC_COEFF * upper))) * upper
        lower = (1 / (1 + np.exp(-LOGISTIC_COEFF * lower))) * lower
        
        return (abs(np.sum((upper - lower) * self.samples[:,0])) 
                    / self.mean.shape[0])
        
    def update_prior_search(self, model, num_samples=20000):
        # only need to update the model
        self.model = model
        self.samples = self.inputs.samples[np.random.choice(
                                        np.arange(self.inputs.samples.shape[0]), 
                                        num_samples)]
        self.mean, self.std = self.model.predict(self.samples, 
                                                 return_std=True)

    def compute_reduced_std(self, x):
        '''based on iterative Gaussian process Eq.19-20'''
        K_trans_x = self.model.kernel_(x, self.model.X_train_)
        K_trans_mc = self.model.kernel_(self.samples,
                                          self.model.X_train_)
        
        V_x = solve_triangular(
                self.model.L_, K_trans_x.T, lower=True, check_finite=False)
        V_mc = solve_triangular(
                self.model.L_, K_trans_mc.T, lower=True, check_finite=False)
        cov_x_mc = self.model.kernel_(x, self.samples) - V_x.T @ V_mc
        var_reduction = cov_x_mc ** 2 / ((self.model.predict(x, 
                                            return_std=True)[1])**2)
        
        reduced_std = np.sqrt(self.std**2 - var_reduction)
        
        return reduced_std