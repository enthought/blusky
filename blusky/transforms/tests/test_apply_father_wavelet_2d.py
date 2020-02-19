from os import path

from keras.models import Model
from keras.layers import Input
import numpy as np
from scipy.signal import convolve2d
import unittest

from blusky.transforms.cascade_2d import Cascade2D
from blusky.wavelets.wavelet_factories_2d import (
    vanilla_morlet_2d,
    vanilla_gabor_2d,
)
from blusky.utils.pad_2d import Pad2D
from blusky.transforms.apply_father_wavelet_2d import ApplyFatherWavlet2D
from blusky.transforms.default_decimation import NoDecimation
from blusky.transforms.transform_factories_2d import (
    vanilla_scattering_transform,
)

import blusky.datasets as datasets
from blusky.transforms.cascade_tree import CascadeTree


class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        """
        Create a 2d cascade with three wavelets and order 3, and compare
        results with manual convolution.
        """
        test_file_1 = path.join(
            path.dirname(datasets.__file__), "twod_image_1.npy"
        )

        original_image = np.load(test_file_1)

        # get a single tile from the image to test
        # note this image is currently unpadded.
        # how many boundary elements are needed to pad?
        extracted_image = original_image[0:32, 0:32]

        self.img = np.expand_dims(extracted_image, axis=-1)

        # Don't make this too huge for brevity.
        self.J = 3
        # 0 = no overlap etc.
        self.overlap_log_2 = 0
        # apply to all available orders
        self.order = 3
        # Should be one or more to avoid aliasing, if you want overlapping
        # tiles this can increase too.
        self.oversampling = 1

        self.num_angles = 3
        self.angles = tuple(
            [
                90.0
                - np.rad2deg(
                    (int(self.num_angles - self.num_angles / 2 - 1) - theta)
                    * np.pi
                    / self.num_angles
                )
                for theta in range(self.num_angles)
            ]
        )

        # details of the input data
        self.sample_rate = 0.004 * 3

        # vanilla filter bank
        wavelets = [
            vanilla_morlet_2d(self.sample_rate, j=i) for i in range(0, self.J)
        ]
        father_wavelet = vanilla_gabor_2d(self.sample_rate, j=self.J)

        # extract the kernels of each of the wavelets for manual convolution
        # we'll test using three different angles that we used to create the
        # transform above.
        wav1 = wavelets[0]
        wav2 = wavelets[1]
        wav3 = wavelets[2]

        # extract the kernels of each of the wavelets for manual convolution
        # we'll test using three different angles that we used to create the
        # transform above.
        wav1_k = wav1.kernel(self.angles[0])
        wav2_k = wav2.kernel(self.angles[1])
        wav3_k = wav3.kernel(self.angles[2])

        phi = father_wavelet.kernel(0.0)

        npad = 31
        img_pad = np.pad(
            self.img, ((npad, npad), (npad, npad), (0, 0)), mode="reflect"
        )
        # get numpy array of the test input image
        x = img_pad[:, :, 0]

        # manual convolution, |x * psi_1|
        conv = np.abs(convolve2d(x, wav1_k, mode="same"))
        conv2 = np.abs(convolve2d(conv, wav2_k, mode="same"))
        conv3 = np.abs(convolve2d(conv2, wav3_k, mode="same"))

        # unpad the original image, and convolve with the phi
        # note that the dimensions for phi are one less than the
        # conv result, so we get a 4x4 result.  Take the first one
        self.manual_result1 = convolve2d(
            conv[npad:-npad, npad:-npad], phi.real, mode="valid"
        )[0, 0]
        self.manual_result2 = convolve2d(
            conv2[npad:-npad, npad:-npad], phi.real, mode="valid"
        )[0, 0]
        self.manual_result3 = convolve2d(
            conv3[npad:-npad, npad:-npad], phi.real, mode="valid"
        )[0, 0]

    def test_apply_father_wavelet_results(self):
        # Complete the scattering transform with the father wavelet
        father_wavelet = vanilla_gabor_2d(self.sample_rate, j=self.J)
        wavelets = [
            vanilla_morlet_2d(self.sample_rate, j=i) for i in range(0, self.J)
        ]

        apply_conv = ApplyFatherWavlet2D(
            J=self.J,
            overlap_log_2=self.overlap_log_2,
            img_size=self.img.shape,
            sample_rate=self.sample_rate,
            wavelet=father_wavelet,
        )

        deci = NoDecimation()  # DefaultDecimation(oversampling=oversampling)

        # input
        inp = Input(shape=self.img.shape)

        # valid padding
        cascade2d = Cascade2D("none", 0, decimation=deci, angles=self.angles)

        # Pad the input
        pad_2d = Pad2D(wavelets, decimation=deci)
        padded = pad_2d.pad(inp)

        # Apply cascade with successive decimation
        cascade_tree = CascadeTree(padded, order=self.order)
        cascade_tree.generate(wavelets, cascade2d._convolve)
        convs = cascade_tree.get_convolutions()

        # Create layers to remove padding
        cascade_tree = CascadeTree(padded, order=self.order)
        cascade_tree.generate(wavelets, pad_2d._unpad_same)
        unpad = cascade_tree.get_convolutions()

        # Remove the padding
        unpadded_convs = [i[1](i[0]) for i in zip(convs, unpad)]

        sca_transf = apply_conv.convolve(unpadded_convs)

        model = Model(inputs=inp, outputs=sca_transf)

        result = model.predict(np.expand_dims(self.img, axis=0))

        # get cnn result, note we're using the third angle for this (index 2)
        cnn_result1 = result[0][0, 0, 0, 0]
        cnn_result2 = result[3][0, 0, 0, 1]
        cnn_result3 = result[6][0, 0, 0, 5]

        # use all close to assert relative error:
        manual = np.array(
            [self.manual_result1, self.manual_result2, self.manual_result3]
        )
        manual[1:] /= manual[0]

        cnn_result = np.array([cnn_result1, cnn_result2, cnn_result3])
        cnn_result[1:] /= cnn_result[0]

        np.testing.assert_allclose(
            manual,
            cnn_result,
            atol=1e-3,
            err_msg="first order does not match with cnn result.",
        )

    def test_vanilla_wavelet(self):
        N, M, _ = self.img.shape

        model, _ = vanilla_scattering_transform(
            self.J,
            overlap_log_2=self.overlap_log_2,
            img_size=self.img.shape,
            sample_rate=self.sample_rate,
            num_angles=self.num_angles,
            order=self.order,
        )
        
        result = model.predict(np.expand_dims(self.img, axis=0))

        cnn_result1 = result[0][0, 0, 0, 0]
        cnn_result2 = result[3][0, 0, 0, 1]
        cnn_result3 = result[6][0, 0, 0, 14]
        
        # use all close to assert relative error:
        manual = np.array(
            [self.manual_result1, self.manual_result2, self.manual_result3]
        )
        manual[1:] /= manual[0]

        cnn_result = np.array([cnn_result1, cnn_result2, cnn_result3])
        cnn_result[1:] /= cnn_result[0]

        np.testing.assert_allclose(
            manual,
            cnn_result,
            atol=1e-3,
            err_msg="first order does not match with cnn result.",
        )

    def test_apply_father_wavelet_dirac(self):
        """
        Test that a Dirac function input returns the original wavelet.
        """
        pass
