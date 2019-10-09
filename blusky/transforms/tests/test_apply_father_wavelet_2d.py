from os import path

from keras.models import Model
from keras.layers import Input
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import unittest

from blusky.wavelets.morlet2d import Morlet2D
from blusky.transforms.cascade_2d import Cascade2D
from blusky.wavelets.wavelet_factories_2d import (
    vanilla_morlet_2d,
    vanilla_gabor_2d,
)
from blusky.utils.pad_2d import Pad2D
from blusky.transforms.apply_father_wavelet_2d import ApplyFatherWavlet2D
from blusky.transforms.default_decimation import NoDecimation
import blusky.datasets as datasets
from blusky.transforms.cascade_tree import CascadeTree


class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        # define the test cascade. use
        self.cascade = Cascade2D("none", 0, angles=(0.0, 45, 90))

    def test_apply_father_wavelet_results(self):
        """
        Create a 2d cascade with three wavelets and order 3, and compare
        results with manual convolution.
        """
        test_file_1 = path.join(
            path.dirname(datasets.__file__), "twod_image_1.npy"
        )

        original_image = np.load(test_file_1)
        original_shape = original_image.shape

        # get a single tile from the image to test
        # note this image is currently unpadded.
        # how many boundary elements are needed to pad?
        extracted_image = original_image[0:32, 0:32]
        extracted_shape = extracted_image.shape

        img = np.expand_dims(extracted_image, axis=-1)

        # Don't make this too huge for brevity.
        J = 3
        # 0 = no overlap etc.
        overlap_log_2 = 0
        # apply to all available orders
        order = 3
        # Should be one or more to avoid aliasing, if you want overlapping tiles,
        # this can increase too.
        oversampling = 0

        angles = (0.0, 45.0, 90.0)

        # details of the input data
        img_size = img.shape
        sample_rate = 0.004 * 3

        # vanilla filter bank
        wavelets = [vanilla_morlet_2d(sample_rate, j=i) for i in range(0, J)]
        father_wavelet = vanilla_gabor_2d(sample_rate, j=J)
        print(father_wavelet.kernel(0.0).shape)
        # method of decimation
        deci = NoDecimation()  # DefaultDecimation(oversampling=oversampling)

        # input
        inp = Input(shape=img.shape)

        # valid padding
        cascade2d = Cascade2D("none", 0, decimation=deci, angles=angles)

        # Pad the input
        pad_2d = Pad2D(wavelets, decimation=deci)
        padded = pad_2d.pad(inp)

        # Apply cascade with successive decimation
        cascade_tree = CascadeTree(padded, order=order)
        cascade_tree.generate(wavelets, cascade2d._convolve)
        convs = cascade_tree.get_convolutions()

        # Create layers to remove padding
        cascade_tree = CascadeTree(padded, order=order)
        cascade_tree.generate(wavelets, pad_2d._unpad_same)
        unpad = cascade_tree.get_convolutions()

        # Remove the padding
        unpadded_convs = [i[1](i[0]) for i in zip(convs, unpad)]

        # Complete the scattering transform with the father wavelet
        apply_conv = ApplyFatherWavlet2D(
            J=J,
            overlap_log_2=overlap_log_2,
            img_size=img.shape,
            sample_rate=sample_rate,
            wavelet=father_wavelet,
        )

        sca_transf = apply_conv.convolve(unpadded_convs)

        model = Model(inputs=inp, outputs=sca_transf)

        result = model.predict(np.expand_dims(img, axis=0))

        # extract the kernels of each of the wavelets for manual convolution
        # we'll test using three different angles that we used to create the
        # transform above.
        wav1 = wavelets[0]
        wav2 = wavelets[1]
        wav3 = wavelets[2]

        # extract the kernels of each of the wavelets for manual convolution
        # we'll test using three different angles that we used to create the
        # transform above.
        wav1_k = wav1.kernel(0.0)
        wav2_k = wav2.kernel(45.0)
        wav3_k = wav3.kernel(90.0)

        phi = father_wavelet.kernel(0.0)

        img_pad = np.pad(img, ((16, 16), (16, 16), (0, 0)), mode="reflect")
        # get numpy array of the test input image
        x = img_pad[:, :, 0]

        # manual convolution, |x * psi_1|
        conv = np.abs(convolve2d(x, wav1_k, mode="same"))
        conv2 = np.abs(convolve2d(conv, wav2_k, mode="same"))
        conv3 = np.abs(convolve2d(conv2, wav3_k, mode="same"))

        # unpad the original image, and convolve with the phi
        # note that the dimensions for phi are one less than the
        # conv result, so we get a 4x4 result.  Take the first one
        manual_result1 = convolve2d(
            conv[16:-16, 16:-16], phi.real, mode="valid"
        )[0, 0]
        manual_result2 = convolve2d(
            conv2[16:-16, 16:-16], phi.real, mode="valid"
        )[0, 0]
        manual_result3 = convolve2d(
            conv3[16:-16, 16:-16], phi.real, mode="valid"
        )[0, 0]
        # get cnn result, note we're using the third angle for this (index 2)
        cnn_result1 = result[0][0, 0, 0, 0]
        cnn_result2 = result[3][0, 0, 0, 1]
        cnn_result3 = result[6][0, 0, 0, 5]

        np.testing.assert_almost_equal(
            manual_result1,
            cnn_result1,
            err_msg="|x * psi_1| * phi does not match with cnn result.",
        )

        np.testing.assert_almost_equal(
            manual_result2,
            cnn_result2,
            err_msg="||x * psi_1| * psi_2| * phi does not match with cnn result.",
        )

        np.testing.assert_almost_equal(
            manual_result3,
            cnn_result3,
            err_msg="|||x * psi_1| * psi_2| * psi_3| * phi does not match with cnn result.",
        )

    def test_apply_father_wavelet_dirac(self):
        """
        Test that a Dirac function input returns the original wavelet.
        """
        pass
