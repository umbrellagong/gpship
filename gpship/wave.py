import numpy as np
from scipy.signal import hilbert
import scipy.integrate as integrate


pi = np.pi
g = 9.81

class LinearRandomWave():
    '''
    Attributes
    ----------
    Hs: float
        significant wave height
    Tp: float
        peak wave period
    wp: float
        peak wave frequence
    Lp: float
        peak wave length
    kp: float
        peak wave number
    gamma: float
        spectrum band parameter
    whether_Gaussian: bool
        specify Gaussian or Jonswap spectrum
    
    '''
    def __init__(self, Hs=12, Tp=15, gamma=0.02, whether_Gaussian=True):
        self.Hs = Hs
        self.Tp = Tp
        self.wp = 2 * pi / Tp
        self.kp = self.wp**2 / g
        self.Lp = 2 * pi / self.kp
        self.whether_Gaussian = whether_Gaussian
        self.gamma = gamma
    
    def prepare_wave(self, seed, base_scale=256, num_mode=1024, 
                         whether_temporal=True):
        # specify spectrum form
        if self.whether_Gaussian:
            alpha = (self.Hs / 4)**2 / (self.gamma * np.sqrt(2 * pi))
            S = self._spectrum_gaussian
        else:
            integration = integrate.quad(self._spectrum_jonswap_single, 
                                         0, 100 * self.wp, 
                                         args=(1, self.wp, self.gamma))[0]
            alpha = (self.Hs / 4) **2 / integration
            S = self._spectrum_jonswap
        # specify random phase
        np.random.seed(seed)
        self.random_phase = np.atleast_2d(np.random.rand(num_mode) * 2*pi).T
        # specify amplitude
        W = np.atleast_2d(np.arange(1, num_mode + 1)).T
        if whether_temporal:
            base = self.wp / base_scale    # frequence base
            self.Amplitude = np.sqrt(2 * S(W * base, alpha, self.wp, self.gamma) 
                                       * base)
            self.period = self.Tp * base_scale
        else:
            base = self.kp / base_scale    # wavenumber base
            self.Amplitude = np.sqrt(g / np.sqrt(g * W * base) 
                                     * S(np.sqrt(g * W * base), alpha, 
                                         self.wp, self.gamma)
                                     * base) 
            self.period = self.Lp * base_scale
        self.num_mode = num_mode
        self.base = base

    def generate_wave(self, t=None, num_points=2048, whether_envelope=False):
        W = np.atleast_2d(np.arange(1, self.num_mode + 1)).T
        
        if t is None:
            t = np.linspace(0, self.period, num_points, endpoint=False)

        temp = self.Amplitude * np.cos(W*t*self.base + self.random_phase)
        
        if np.size(t) != 1:
            elev = np.sum(temp, axis=0)
            if whether_envelope:
                return elev, np.abs(hilbert(elev))
            else:
                return elev
        else:
            return np.sum(temp)

    def _spectrum_gaussian(self, W, alpha, wp, gamma):
        '''
        W is a one-d array
        '''
        return alpha * np.exp(- (W - wp)**2 / (2 * gamma**2))

    def _spectrum_jonswap(self, W, alpha, wp, gamma): 
        return np.array([self._spectrum_jonswap_single(w, alpha, wp, gamma) 
                         for w in W])

    def _spectrum_jonswap_single(self, w, alpha, wp, gamma): 
        if w <= wp:
            return (alpha * g ** 2 / w **5 * np.exp(-1.25 * wp**4 / w**4) 
                   * gamma ** np.exp(- (w-wp)**2 / (2 * 0.07 **2 * wp**2)))
        else:
            return (alpha * g ** 2 / w **5 * np.exp(-1.25 * wp**4 / w**4) 
                    * gamma ** np.exp(- (w-wp)**2 / (2 * 0.09 **2 * wp**2)))
                    
                    
class GroupWave(object):   
    'a wave group with Gaussian-like envelope' 
    def __init__(self, group_amplitude, group_length, group_phase, 
                 wave_period=15):
        '''
        specify the parameters used to generate the wave group, including the 
        group length L, group amplitude A, group phase P, carry wave period T.
        '''
        self.A = group_amplitude
        self.L = group_length
        self.P = group_phase
        self.T = wave_period
    
    def prepare_wave(self):
        pass
    
    def generate_wave(self, t):
        '''
        Generating the wave group. The envelope of the group is centered at t=0
        '''
        return (self.A * np.exp(-t**2 / ( 2 * self.L**2))
                * np.cos(2 * pi/self.T * t + self.P))