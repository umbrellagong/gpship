import numpy as np
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed


EXPLODE_THRESHOLD = 10


def compute_ship_roll(wave_signal, compute_range, init_cond=None, 
                      max_step=0.5, t_eval=None, print_interval=0.1,
                      alpha1=0.35, alpha2=0.06, 
                      beta1=0.04, beta2=-0.1, 
                      ep1=0.016 * np.cos(np.pi / 6), 
                      ep2=0.012 * np.sin(np.pi / 6)):
    '''Solve the roll equation for ship response '''
    if init_cond is None:
            init_cond = [0, 0]
    
    if t_eval is None:
            t_eval = np.arange(compute_range[0], compute_range[1], 
                               print_interval)

    return solve_ivp(ode_rhs, compute_range, init_cond, method='RK45', 
                     t_eval=t_eval, max_step=max_step, events=explode,
                     args=(wave_signal, alpha1, alpha2, beta1, beta2, ep1, ep2))


def ode_rhs(t, y, wave_signal, alpha1, alpha2, beta1, beta2, ep1, ep2):
    '''The right hand side of the ship roll equation'''
    y0d = (- alpha1 * y[0] - alpha2 * y[0] * np.abs(y[0]) 
          - (beta1 + ep1 * wave_signal(t)) * y[1] - beta2 * y[1]**3 
          + ep2 * wave_signal(t))
    y1d = y[0]
    return [y0d, y1d]


def ivp_event(event_func):
    event_func.terminal = True
    event_func.direction = 1
    return event_func
 

@ivp_event                  # explode function 
def explode(t, y, wave_signal, alpha1, alpha2, beta1, beta2, ep1, ep2): 
    return abs(y[1]) - EXPLODE_THRESHOLD
