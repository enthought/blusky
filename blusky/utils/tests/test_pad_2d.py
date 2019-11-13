from os import path
import unittest

from keras.layers import Input
from keras.models import Model
import numpy as np
from PIL import Image

import blusky.datasets as datasets
from blusky.transforms.cascade_2d import Cascade2D
from blusky.transforms.default_decimation import DefaultDecimation
from blusky.utils.pad_2d import Pad2D
from blusky.wavelets.wavelet_factories_2d import vanilla_morlet_2d


class TestPad2D(unittest.TestCase):
    def setUp(self):
        """Test Pad2d"""
        self.wavelets = [
            vanilla_morlet_2d(0.004, j=0),
            vanilla_morlet_2d(0.004, j=1),
            vanilla_morlet_2d(0.004, j=2),
        ]

        self.deci = DefaultDecimation(oversampling=1)

        self.cascade = Cascade2D(
            "none", 0, decimation=self.deci, angles=(0.0, 45.0, 90.0)
        )

        self.pad_2d = Pad2D(self.wavelets, decimation=self.deci)

        imgs = []
        for label in ["chaotic", "fault", "salt", "horizon"]:
            for im in range(0, 3):
                data = path.join(
                    path.join(path.dirname(datasets.__file__), "test_tiles"),
                    "{}{}.png".format(label, im),
                )
                data = np.array(Image.open(data)).astype(np.float32)
                data /= np.max(data)
                imgs.append(data)
        self.imgs = np.array(imgs)
        self.imgs.shape = list(self.imgs.shape) + [1]

    def test_pad(self):
        inp = Input(shape=(99, 129, 1))
        padded = self.pad_2d.pad(inp)
        self.assertTrue(padded.shape[1] == 131)
        self.assertTrue(padded.shape[2] == 161)

    def test_unpad_1(self):
        from blusky.transforms.cascade_tree import CascadeTree

        inp = Input(shape=(99, 99, 1))
        padded = self.pad_2d.pad(inp)

        cascade_tree = CascadeTree(padded, order=2)
        cascade_tree.generate(self.wavelets, self.cascade._convolve)
        convs = cascade_tree.get_convolutions()

        cascade_tree = CascadeTree(padded, order=2)
        cascade_tree.generate(self.wavelets, self.pad_2d.unpad)
        unpadded = cascade_tree.get_convolutions()

        unpadded_convs = [i[1](i[0]) for i in zip(convs, unpadded)]

        # The reason the shape is changing is because its being decimated.

        self.assertTrue(unpadded_convs[0].shape[1] == 99)
        self.assertTrue(unpadded_convs[0].shape[2] == 99)

        self.assertTrue(unpadded_convs[1].shape[1] == 50)
        self.assertTrue(unpadded_convs[1].shape[2] == 50)

        self.assertTrue(unpadded_convs[2].shape[1] == 25)
        self.assertTrue(unpadded_convs[2].shape[2] == 25)

        self.assertTrue(unpadded_convs[3].shape[1] == 50)
        self.assertTrue(unpadded_convs[3].shape[2] == 50)

        self.assertTrue(unpadded_convs[4].shape[1] == 25)
        self.assertTrue(unpadded_convs[4].shape[2] == 25)

        self.assertTrue(unpadded_convs[5].shape[1] == 25)
        self.assertTrue(unpadded_convs[5].shape[2] == 25)

    def test_unpad_2(self):
        from blusky.transforms.cascade_tree import CascadeTree

        # --- Pad/Unpad --- #
        inp = Input(shape=(99, 99, 1))
        padded = self.pad_2d.pad(inp)

        cascade_tree = CascadeTree(padded, order=2)
        cascade_tree.generate(self.wavelets, self.cascade._convolve)
        convs = cascade_tree.get_convolutions()

        cascade_tree = CascadeTree(padded, order=2)
        cascade_tree.generate(self.wavelets, self.pad_2d.unpad)
        unpadded = cascade_tree.get_convolutions()

        unpadded_convs = [i[1](i[0]) for i in zip(convs, unpadded)]

        model_1 = Model(inputs=inp, outputs=unpadded_convs)

        result_1 = model_1.predict(self.imgs[:1])

        # --- Manually --- #
        inp = Input(shape=(131, 131, 1))

        cascade_tree = CascadeTree(inp, order=2)
        cascade_tree.generate(self.wavelets, self.cascade._convolve)
        convs = cascade_tree.get_convolutions()

        model_2 = Model(inputs=inp, outputs=convs)

        padded = np.pad(
            np.array(self.imgs[:1]),
            ((0, 0), (16, 16), (16, 16), (0, 0)),
            "reflect",
        )

        result_2 = model_2.predict(padded)

        result_2[0] = result_2[0][:, 16:-16, 16:-16, :]

        np.allclose(result_1[0], result_2[0])
