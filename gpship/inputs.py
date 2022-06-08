import numpy as np
from pyDOE import lhs
from wave import LinearRandomWave 
from joblib import Parallel, delayed


class Inputs():
    '''The wave input space (L, A) after parameterization '''
    def __init__(self, dim, domain):
        self.dim = dim
        self.domain = domain

    def sampling(self, num, criterion=None):
        # for initial sampling
        lhd = lhs(self.dim, num, criterion=criterion)
        lhd = self.rescale_samples(lhd, self.domain)
        return lhd
        
    def load_samples(self, path, num_seeds=10000, length=3840):
        self.samples = np.load(path, allow_pickle=True)
        self.num_seeds = num_seeds
        self.length = length
            
    @staticmethod
    def rescale_samples(x, domain):
        """Rescale samples from [0,1]^d to actual domain."""
        for i in range(x.shape[1]):
            bd = domain[i]
            x[:,i] = x[:,i]*(bd[1]-bd[0]) + bd[0]
        return x

def prepaer_inputs(threshold=9.6, num=10000):
    ''' Set a physical bar for wave amplitude. 
    
    Delete the wave field with groups above the bar. 
    '''
    GroupPool = np.load('wavedata/GroupPool.npy', allow_pickle=True)
    samples = []
    scores = []
    seeds = []
    seed_num = 0
    for seed, i in enumerate(GroupPool):
        if len(i) != 1:
            if np.max(np.array(i)[:,2]) > threshold:
                continue
            
            for j in i:
                samples.append([j[1], j[2]])
                scores.append(j[3])
            seeds.append(seed)
            seed_num = seed_num + 1
            if seed_num == num:
                break
    
    np.save('wavedata/scores', scores)
    np.save('wavedata/samples', samples)
    np.save('wavedata/seeds', seeds)

    
def convert_groups():
    '''Generate unnormalized GroupPool from raw detection with normalized 
    length.'''
    # Keep the dimensional value 
    GroupPool = np.load('wavedata/raw_detection.npy', 
                        allow_pickle=True).item()['detections']
    for i in GroupPool:
        if len(i) != 1:
            for j in i:
                #j[0] = j[0] * 1.875 + 1920
                j[1] = j[1] * 1.875
    np.save('wavedata/GroupPool.npy', GroupPool)
    
    
if __name__ == '__main__':
    convert_groups()
    prepaer_inputs(12)
    
    