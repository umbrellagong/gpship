import numpy as np
from wave import LinearRandomWave
from ship import compute_ship_roll
from joblib import Parallel, delayed


def find_starting(waveField, before=0, start=200, length=300, step=30,
                             threshold=4):
    t = start
    end = start + length
    ele_list = [9999]     
    while (t < end) & (max(ele_list) > threshold):
        t_list = np.arange(before - (t + step), 
                            before - t, 0.1)
        ele_list = abs(waveField.generate_wave(t_list))
        t = t + step
    starting_t = t_list[np.argmin(ele_list)]
    return starting_t
    
    
def compute_response_single(seed, kws_wave_spectrum, kws_wave_field, kws_ship):
    '''compute the ship roll in one wave realization'''
    
    # generate the wave field
    waveField = LinearRandomWave(**kws_wave_spectrum)
    waveField.prepare_wave(seed, **kws_wave_field)
    # put the wave field into roll equation
    starting_t = find_starting(waveField)
    computeRange = [starting_t, waveField.period]
    t_eval = np.arange(0, waveField.period, 0.1) 

    try:
        SolverResults = compute_ship_roll(waveField.generate_wave,
                                          computeRange, 
                                          t_eval=t_eval,
                                          **kws_ship)
    except Exception:
        print('The problem is at Index: ', seed)
        return -2
    if SolverResults.status != 0:
        print("explode in Index", seed)
        return -1
    else:
        return SolverResults.y[1]


def compute_response_batch(num_p=20, kws_wave_spectrum={}, kws_wave_field={}, 
                           kws_ship={}):
    '''compute the ship roll in a large number of wave realizations'''
    seeds = np.load('wavedata/seeds.npy', allow_pickle=True)
    results = Parallel(n_jobs=num_p)(delayed(compute_response_single)
                                        (seed, kws_wave_spectrum, 
                                        kws_wave_field, kws_ship) 
                                        for seed in seeds)
    np.save('results/true', results)
    

if __name__ == "__main__":
    num_p = 20
    kws_wave_spectrum = {}
    kws_wave_field = {}
    kws_ship = {}
    compute_response_batch(num_p, kws_wave_spectrum, kws_wave_field, kws_ship)
                               
                               
                               
                               
                               
                               
                               
                               
                               
                               