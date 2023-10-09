import numpy as np
import sys

def ccc(x,y, ignore=-5.0):
    """
        y_true: shape of (N, )
        y_pred: shape of (N, )
        """

    #y = y.reshape(-1)
    #x = x.reshape(-1)
    #index = y != ignore
    #y = y[index]
    #x = x[index]

    if len(y) <= 1:
        sys.exit()
        return 0.0
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)) +1e-8)
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/((x_s**2 + y_s**2 + (x_m - y_m)**2)+ 1e-8)
    return ccc