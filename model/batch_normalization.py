import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x = np.array(x)
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)
        if training:
            m = np.mean(x, axis=0)
            v = np.var(x, axis=0)
            xhat = (x - m) / np.sqrt(v+1e-7) 
            running_mean = (1-momentum)*running_mean + momentum*m 
            running_var = (1-momentum)*running_var + momentum*v
        else:
            xhat = (x-running_mean) / (np.sqrt(running_var + eps))
        out = gamma*xhat + beta
        return (np.round(out,4), np.round(running_mean,4), np.round(running_var,4))
