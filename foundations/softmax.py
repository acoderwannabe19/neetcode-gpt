import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        z_shifted = z - max(z)
        exp = np.exp(z_shifted) 
        s = exp / np.sum(exp)
        return np.round(s, 4)
        
