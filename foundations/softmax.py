import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        z -= np.max(z)
        exps = np.exp(z)
        return np.round(exps / (np.sum(exps)), 4)
