import numpy as np
from tensorflow import convert_to_tensor
from tensorflow.keras.initializers import Initializer

class BluskyNetWeights1D(Initializer):
    def __init__(self, wav):
        """                
        Create an initializer for Conv1D layers.

        Parameters
        ----------
        wavelet1d - Array
            The wavelet.

        Returns
        -------
        returns - tensorflow variable
            returns a tensorflow variable containing the weights.
        """      
        self.wav = wav

    def __call__(self, shape, dtype=None):
        # input(batch, nsample, nfeatures)
        # Conv1D(nfilters, kernel_size)
        # shape = (kernel_size, nfeatures, nfilters)        
        weights = np.zeros(shape)

        # apply to each input channel
        for i in range(shape[1]):
            for j in range(shape[2]):
                weights[:, i, j] = self.wav.astype(np.float32)

        return convert_to_tensor(weights, dtype=dtype)

