import copy
import numpy as np
from scipy import optimize
from sklearn.base import clone
from joblib import Parallel, delayed


class OptimalDesign():
    ''' BED class for temporal exceeding probability
    
    Parameters
    -----------
    f: func
        The response function
    input: instance of Inputs class
        Include dimensions, bounds, and samples.
    
    Attributes
    ----------
    DX : array (n_samples, n_dim)
        The input of the samples.
    DY : array (n_samples,)
        The output of the samples.
    '''
    
    def __init__(self, f, inputs, threshold):
        self.f = f    # The response function  
        self.inputs = inputs
        self.threshold = threshold
        
    def init_sampling(self, n_init): 
        '''Generate initial samples.
        
        Parameters
        ----------
        n_init: int
            The number of initial samples.
        '''
        self.DX = self.inputs.sampling(n_init)
        self.DY = self.f(self.DX, self.threshold)   # a vector
        return self
    
    def seq_sampling(self, n_seq, acq, model, num_grid=40, n_jobs=1):
        '''Generate sequential samples. 
        
        Parameters
        ----------
        n_seq: int
            The number of sequential samples.
        acq: instance of Acq class
            Represent the design of acq (or the objective of problem).
        model: instance of GaussianProcessRegressor
            The learned Gaussian process regressor from samples.
        num_grid: int
            Number of grids in brute-force optimizer.
        n_jobs: int
            The number of workers used by brute-force optimizer.
            
        Return
        ----------
        exceeding_list: list
            The exceeding probabilities after adding each 
            sequential samples.
        
        DX: array
            The final input. 
        '''
        
        self.acq = copy.copy(acq)
        self.model = clone(model)
        exceeding_list = []
        
        exceeding = self.fit_comp(self.model, self.DX, self.DY, self.inputs)
        exceeding_list.append(exceeding)
        
        for i in range(n_seq):
            
            self.acq.update_prior_search(self.model)
            opt_pos = optimize.brute(self.acq.compute_value, self.inputs.domain,
                                    Ns=num_grid, full_output=False, finish=None,
                                    workers=n_jobs)
                        
            self.DX = np.append(self.DX, np.atleast_2d(opt_pos), axis=0)
            self.DY = np.append(self.DY, self.f(opt_pos, self.threshold))
            print('pos: ', self.DX[-1], '  prob:', self.DY[-1])
                        
            exceeding = fit_comp(self.model, self.DX, self.DY, self.inputs)
            exceeding_list.append(exceeding)
        return exceeding_list, self.DX

        
def compute_lh_results(f, threshold, inputs, model, n_init, n_seq):
    '''LH sampling results for comparison'''
    
    model = copy.deepcopy(model)
    exceeding_list = []
    for i in range(n_init, n_init + n_seq + 1):
        DX = inputs.sampling(i)
        DY = f(DX, threshold)
        exceeding = fit_comp(model, DX, DY, inputs)
        exceeding_list.append(exceeding)
    return exceeding_list


def fit_comp(model, DX, DY, inputs):
    '''fit the GP with samples and compute the exceeding prob'''
    model.fit(DX, DY)
    exceeding = model.predict(inputs.samples)
    exceeding = sum(np.where(exceeding < 0, 0, exceeding)
            * inputs.samples[:,0]) / (inputs.num_seeds * inputs.length)
    return exceeding