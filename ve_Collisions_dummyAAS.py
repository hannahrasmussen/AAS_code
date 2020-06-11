import numpy as np
import numba as nb

GF = 1.166e-11
me = 0.511

eps_values, w_values = np.polynomial.laguerre.laggauss(10)

@nb.jit(nopython = True)
def f_elpos(x,T):
    Ep = np.sqrt(x**2 + me**2 / T**2)
    return x**2 * np.exp(x) / (np.exp(Ep)+1)

@nb.jit(nopython = True)
def num_elpos(T):
    return np.sum(w_values * f_elpos(eps_values, T)) * T**3 / np.pi**2

@nb.jit(nopython = True)
def f_eq(p,T):
    return 1.0/(np.exp(np.sqrt(p**2+me**2)/T)+1)

@nb.jit(nopython = True)
def driver(p_array, T, f):
    nsigmav = GF**2 * p_array**2 * num_elpos(T)
    return - nsigmav * (f - f_eq(p_array,T))

@nb.jit(nopython = True)
def cI(i, f, p, T):
    nsigmav = GF**2 * p[i]**2 * num_elpos(T)
    return - nsigmav * (f[i] - f_eq(p[i],T))
