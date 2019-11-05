from os import path

from keras.models import Model
from keras.layers import Input
import numpy as np
from scipy.signal import convolve
import unittest


import blusky.datasets as datasets
from blusky.transforms.default_decimation import NoDecimation
from blusky.transforms.cascade_1d import Cascade1D
from blusky.transforms.cascade_tree import CascadeTree
from blusky.utils.pad_1d import pad_to_log2, Pad1D
from blusky.wavelets.wavelet_factories_1d import (vanilla_gabor_1d,
                                                  vanilla_morlet_1d)


class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        # define the test cascade.
        """
        Create a 1d cascade with three wavelets and order 3, and compare
        results with manual convolution.
        """

        self.J = 5
        self.N = 128
        self.order = 3
        self.sample_rate = 1.0

        ts_path = path.join(
            path.dirname(datasets.__file__), "timeseries.csv"
        )
        my_data = np.genfromtxt(ts_path, delimiter=',', skip_header=1)
        self.ts = np.expand_dims(my_data[-self.N:,1], axis=-1)
        self.ts /= np.max(self.ts)
        
        # vanilla filter bank
        wavelets = [vanilla_morlet_1d(self.sample_rate, j=i)
                        for i in range(0, self.J)]
        father_wavelet = vanilla_gabor_1d(self.sample_rate, j=self.J)

        # extract the kernels of each of the wavelets for manual convolution
        # we'll test using three different angles that we used to create the
        # transform above.
        wav1_k = wavelets[0].kernel()
        wav2_k = wavelets[1].kernel()
        wav3_k = wavelets[2].kernel()

        x = np.pad(self.ts[:,0], (128,128), mode="reflect")
        print (x.shape)
        
        # manual convolution, |x * psi_1|
        self.conv1 = np.abs(convolve(x, wav1_k, mode="same"))
        self.conv2 = np.abs(convolve(self.conv1, wav2_k, mode="same"))        
        self.conv3 = np.abs(convolve(self.conv2, wav3_k, mode="same"))

        # unpad
        self.conv1 = self.conv1[128:-128]        
        self.conv2 = self.conv2[128:-128]
        self.conv3 = self.conv3[128:-128]

    def test_cascade_1d_results(self):
        # vanilla filter bank
        wavelets = [vanilla_morlet_1d(self.sample_rate, j=i)
                        for i in range(0, self.J)]
        father_wavelet = vanilla_gabor_1d(self.sample_rate, j=self.J)

        deci = NoDecimation()
        inp = Input(shape=(self.N,1))

        # pad
        pad_1d = Pad1D(wavelets, decimation=deci)
        padded = pad_1d.pad(inp)

        #
        cascade_tree = CascadeTree(padded, order=self.order)
        cascade = Cascade1D(decimation=deci)
        convs = cascade.transform(cascade_tree, wavelets=wavelets)

        # Create layers to remove padding
        cascade_tree = CascadeTree(padded, order=self.order)
        cascade_tree.generate(wavelets, pad_1d._unpad_same)
        unpad = cascade_tree.get_convolutions()

        # Remove the padding
        unpadded_convs = [i[1](i[0]) for i in zip(convs, unpad)]
        
        model = Model(inputs=inp, outputs=unpadded_convs)
        
        result = model.predict(np.expand_dims(self.ts, axis=0))


        cnn_result_1 = np.squeeze(result[0])
        cnn_result_2 = np.squeeze(result[5])
        cnn_result_3 = np.squeeze(result[-10])

        np.testing.assert_allclose(
            self.conv1,
            cnn_result_1,
            atol=1E-3,
            err_msg="first order does not match with cnn result.",
        )

        
        np.testing.assert_allclose(
            self.conv2,
            cnn_result_2,
            atol=1E-3,
            err_msg="first order does not match with cnn result.",
        )

        np.testing.assert_allclose(
            self.conv3,
            cnn_result_3,
            atol=1E-3,
            err_msg="first order does not match with cnn result.",
        )

        
