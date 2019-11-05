from os import path

from keras.models import Model
from keras.layers import Input
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import unittest

from blusky.wavelets.morlet2d import Morlet2D
from blusky.transforms.cascade_2d import Cascade2D
import blusky.datasets as datasets


class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        # define the test cascade. use
        self.cascade = Cascade2D("none", 0, angles=(0.0, 45, 90))

    def test_cascade_2d_results(self):
        """
        Create a 2d cascade with three wavelets and order 3, and compare
        results with manual convolution.
        """
        from blusky.transforms.cascade_tree import CascadeTree

        wav1 = Morlet2D(
            sample_rate=0.004,
            center_frequency=90.0,
            bandwidth=(50.0, 25.0),
            crop=3.5,
            taper=False,
        )
        wav2 = Morlet2D(
            sample_rate=0.004,
            center_frequency=45.0,
            bandwidth=(40.0, 20),
            crop=3.5,
            taper=False,
        )
        wav3 = Morlet2D(
            sample_rate=0.004,
            center_frequency=22.5,
            bandwidth=(20, 10.0),
            crop=3.5,
            taper=False,
        )

        # get the test image
        img_path = path.join(
            path.dirname(datasets.__file__), "test_tiles/chaotic0.png"
        )
        data = np.array(Image.open(img_path)).astype(np.float32)
        data /= np.max(data)

        (nh, nw) = data.shape

        # Keras needs the input images to look like a list
        imgs = np.array([data])
        imgs.shape = list(imgs.shape) + [1]

        # define the input layer of the network to have the
        # shape of the input images in the test suite
        inp = Input(shape=(nh, nw, 1))

        cascade_tree = CascadeTree(inp, order=3)
        cascade_tree.generate([wav1, wav2, wav3], self.cascade._convolve)
        my_transform = cascade_tree.get_convolutions()

        # create the transform
        model = Model(inputs=inp, outputs=my_transform)

        # run the image through the transform cascade
        result = model.predict(imgs)

        # extract the kernels of each of the wavelets for manual convolution
        # we'll test using three different angles that we used to create the
        # transform above.
        wav1_k = wav1.kernel(0.0)
        wav2_k = wav2.kernel(45.0)
        wav3_k = wav3.kernel(90.0)

        # get numpy array of the test input image
        image_index = 0
        x = imgs[image_index, :, :, 0]

        # manual convolution, |x * psi_1|
        conv = np.abs(convolve2d(x, wav1_k, mode="same"))

        # the endpoints are the different convolutions in the transform
        # the first endpoint represents |x * psi_1|
        endpoint = 0
        # the psi_1 kernel is at 0 degrees.  This is the first angle
        # which will have index 0
        angle_index = 0

        # get the corresponding convolution result in the cascade
        cnn_conv = result[endpoint][image_index, :, :, angle_index]

        np.testing.assert_almost_equal(
            conv, cnn_conv, err_msg="Convolution does not match test values."
        )

        # test the second order convolution ||x * psi_1| * psi_2|

        conv2 = np.abs(convolve2d(conv, wav2_k, mode="same"))
        # there are three first order convolutions.  The first second
        # order convolution will be the endpoint with index 3
        endpoint = 3
        # this wavelet is at 45deg, the second angle used to build
        # the cascade.
        angle_index = 1

        cnn_conv2 = result[endpoint][image_index, :, :, angle_index]

        np.testing.assert_almost_equal(
            conv2, cnn_conv2, err_msg="Convolution does not match test values."
        )

        # test the third order convolution |||x * psi_1| * psi_2| * psi_3|
        endpoint = 6
        # The third order convolution is done with the 90 deg psi_3.  This is
        # at angle_index 5
        angle_index = 5
        conv3 = np.abs(convolve2d(conv2, wav3_k, mode="same"))
        cnn_conv3 = result[endpoint][image_index, :, :, angle_index]

        np.testing.assert_almost_equal(
            conv3, cnn_conv3, err_msg="Convolution does not match test values."
        )

    def test_cascade_2d_dirac(self):
        """
        Test that a Dirac function input returns the original wavelet.
        """
        from blusky.transforms.cascade_tree import CascadeTree

        wav1 = Morlet2D(
            sample_rate=0.004,
            center_frequency=90.0,
            bandwidth=(50.0, 25.0),
            crop=3.5,
            taper=False,
        )
        wav1_k = wav1.kernel(0.0)

        # Form the Dirac function, just a unit impulse in the center of the
        # image.
        dirac = np.zeros((99, 99))
        dirac[50, 50] = 1.0

        imgs = np.array([dirac])
        imgs.shape = list(imgs.shape) + [1]

        # define the input layer of the network to have the
        # shape of the input images in the test suite
        inp = Input(shape=(99, 99, 1))

        cascade_tree = CascadeTree(inp, order=3)
        cascade_tree.generate([wav1], self.cascade._convolve)
        my_transform = cascade_tree.get_convolutions()

        model = Model(inputs=inp, outputs=my_transform)

        result = model.predict(imgs)

        cnn_conv_dirac = result[0, :, :, 0]
        cnn_conv_dirac_crop = cnn_conv_dirac[44:57, 44:57]

        np.testing.assert_almost_equal(
            np.abs(wav1_k),
            cnn_conv_dirac_crop,
            err_msg="Convolution does not match test values.",
        )
