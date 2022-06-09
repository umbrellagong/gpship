import numpy as np
from wave import GroupWave
from ship import compute_ship_roll
from joblib import Parallel, delayed


def f(x, threshold, num_proc=1, kws_ship={}):
    ''' The h function in paper
    
    x: array (n, 2) or (2, )
        dimensional group parameter (s, m)
    '''
    x = np.array(x)
    if x.ndim == 1:
        return exceeding(x, threshold, kws_ship=kws_ship)
    if x.ndim == 2:
        if num_proc > 1:
            results = Parallel(n_jobs=num_proc)(delayed(exceeding)(i, threshold, 
                                                            kws_ship=kws_ship) 
                                                            for i in x)
            return results    
        else:
            return np.array([exceeding(i, threshold, kws_ship=kws_ship) 
                            for i in x])


def exceeding(x, threshold, num_l=2, Tp=15, phase=0, kws_ship={}):
    ''' The h function in paper
    
    x: array (2,)
        dimensional group parameter (s, m)
    '''
    L, A = x
    groupwave = GroupWave(A, L, phase, Tp)
    solver_result = compute_ship_roll(groupwave.generate_wave, 
                                      [-num_l*L, num_l*L], 
                                      **kws_ship)
    roll_history = solver_result.y[1]
    
    if np.max(abs(roll_history)) > threshold:
            exceeding =  (np.count_nonzero(abs(roll_history) > threshold) 
                     / (roll_history.shape[0] / (2*num_l))) 
    else:
            exceeding  = (np.max(abs(roll_history)) - threshold) / threshold
    
    return exceeding
    
