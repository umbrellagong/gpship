import numpy as np
from wave import LinearRandomWave
from ship import compute_ship_roll
from wave import GroupWave
from joblib import Parallel, delayed


def compute_group_batch(num_proc=20):
    samples = np.load('wavedata/samples.npy', allow_pickle=True)
    results = Parallel(n_jobs=num_proc)(delayed(compute_group_single)(x) 
                                                for x in samples)
    np.save('results/true_groups', results)


def compute_group_single(x, num_l=3, Tp=15, kws_ship={}):

    L, A = x
    groupwave = GroupWave(A, L, 0, Tp)
    solver_result = compute_ship_roll(groupwave.generate_wave, 
                                      [-num_l*L, num_l*L], 
                                      **kws_ship)
    roll_history = solver_result.y[1]
    
    return roll_history


if __name__ == "__main__":
    compute_group_batch()

                               
                               
                               
                               
                               
                               
                               
                               
                               
                               