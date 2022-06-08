import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel as C
from joblib import Parallel, delayed
from exceeding import f
from acquisitions import AcqTemporal
from inputs import Inputs
from optimaldesign import OptimalDesign, compute_lh_results
import warnings

def main():
    '''perform a batch of designs'''
    
    n_trails = 100
    threshold = 0.25  
    #threshold = 0.30   
    
    dim = 2
    domain = [[3, 120], [0, 12]]
    inputs = Inputs(dim, domain)
    # 9500 wave fields with each 3400 s
    inputs.load_samples("wavedata/samples_075.npy", num_seeds=9500)

    kernel = C(10, (1e-1, 1e2)) * RBF((20, 2), ((1e0, 1e2), (1e-1, 1e1))) 

    # sequential details
    n_init=8
    n_seq=52
    
    def one_design(trail):
        warnings.filterwarnings("ignore")
        sgp = GaussianProcessRegressor(kernel, n_restarts_optimizer=6)
        acq = AcqTemporal(inputs)
        np.random.seed(trail)
        opt = OptimalDesign(f, inputs, threshold)
        opt.init_sampling(n_init)
        results = opt.seq_sampling(n_seq, acq, sgp, num_grid=40, n_jobs=1)
        return results
        
    def one_design_lh(trail):
        warnings.filterwarnings("ignore")
        sgp = GaussianProcessRegressor(kernel, n_restarts_optimizer=6)
        np.random.seed(trail)
        results = compute_lh_results(f, threshold, inputs, sgp, n_init, n_seq)
        return results

    lh_results = Parallel(n_jobs=25)(delayed(one_design_lh)(j)
                                     for j in range(n_trails))
                                     
    np.save('results/seq', lh_results)
    
    seq_results = Parallel(n_jobs=25)(delayed(one_design)(j)
                                     for j in range(n_trails))
    np.save('results/lh', seq_results)
    

    
    
if __name__=='__main__':
    main()
    