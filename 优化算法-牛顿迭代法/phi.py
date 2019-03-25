import numpy as np

def phi(m1: float, m2: float) -> float:
    phi = -1*np.power(m1,3)-np.power(m2,3)+3*np.power(m1,2)+2*np.power(m2,2)+m1+m2-1
    return phi
