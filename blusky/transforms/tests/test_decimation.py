import unittest

from keras.layers import Input

from blusky.transforms.cascade_tree import CascadeTree
from blusky.transforms.default_decimation import (
    DefaultDecimation,
    NoDecimation,
)
from traits.api import HasStrictTraits, Int


class TestAlgorithms(unittest.TestCase):
    def test_no_decimation(self):
        """ Test the no decimation model.
        """
        import numpy as np

        deci = NoDecimation()
        wav = np.empty((99, 99))
        wavp = deci.decimate_wavelet(wav, 2)
        np.array_equal(wav, wavp)

        w, c = deci.resolve_scales(None)
        self.assertTrue(w == 1 and c == 1)

    def test_default_decimation(self):
        """ test the default decimation method.

            At the first layer we decimate according to scale.
            At subsequent layers, we decimate according to the difference
            in order with the current and previuous layers.
        """
        import numpy as np

        class test_wav(HasStrictTraits):
            scale = Int(1)

        cascade_tree = CascadeTree(Input(shape=(64, 64, 1)), order=4)
        cascade_tree.generate(
            [
                test_wav(scale=1),
                test_wav(scale=2),
                test_wav(scale=3),
                test_wav(scale=4),
            ],
            lambda x, y, z: x,
        )

        # go down a path in the tree
        root_node = cascade_tree.root_node
        first_layer = root_node.children[0]
        second_layer = first_layer.children[0]
        third_layer = second_layer.children[0]

        deci = DefaultDecimation(oversampling=0)
        wav = np.empty((64, 64))
        wavp = deci.decimate_wavelet(wav, 2)
        np.array_equal(wav[::2, ::2], wavp)

        # if I would decimate wavelet/conv
        deci = DefaultDecimation(oversampling=0)

        w, c = deci.resolve_scales(root_node)
        self.assertTrue(w == 1 and c == 1)
        w, c = deci.resolve_scales(first_layer)
        self.assertTrue(w == 1 and c == 2)
        w, c = deci.resolve_scales(second_layer)
        self.assertTrue(w == 2 and c == 2)
        w, c = deci.resolve_scales(third_layer)
        self.assertTrue(w == 4 and c == 2)

        #
        deci = DefaultDecimation(oversampling=1)

        w, c = deci.resolve_scales(root_node)
        self.assertTrue(w == 1 and c == 1)
        w, c = deci.resolve_scales(first_layer)
        self.assertTrue(w == 1 and c == 1)
        w, c = deci.resolve_scales(second_layer)
        self.assertTrue(w == 1 and c == 2)
        w, c = deci.resolve_scales(third_layer)
        self.assertTrue(w == 2 and c == 2)

        #
        deci = DefaultDecimation(oversampling=2)

        w, c = deci.resolve_scales(root_node)
        self.assertTrue(w == 1 and c == 1)
        w, c = deci.resolve_scales(first_layer)
        self.assertTrue(w == 1 and c == 1)
        w, c = deci.resolve_scales(second_layer)
        self.assertTrue(w == 1 and c == 1)
        w, c = deci.resolve_scales(third_layer)
        self.assertTrue(w == 1 and c == 2)

        deci = DefaultDecimation(oversampling=0)

        root_node = cascade_tree.root_node
        first_layer = root_node.children[0].children[0].children[0]

        # convolve with stride of 2 (scale 1 wavelet)
        # Then convolve again, j2 - j1 is 1 so add a stride of 2, decimate
        # wav by 2
        # to adjust bandwidth to decimated signal
        # Then convolve again  j3 - j2 is 1 so add a stride of 2, decimate
        # wave by 4,
        # we've decimated twice by 2 previously
        w, c = deci.resolve_scales(first_layer)
        self.assertTrue(w == 4 and c == 2)

        deci = DefaultDecimation(oversampling=1)
        # convolve with stride of 1 (scale 1 wavelet (oversampling))
        # Then convolve again, j2 - j1 is 1 so add a stride of 2
        # to adjust bandwidth to decimated signal
        # Then convolve again  j3 - j2 is 1 so add a stride of 2, decimate
        # wave by 2,
        # we've decimated once by 2 previously
        w, c = deci.resolve_scales(first_layer)
        self.assertTrue(w == 2 and c == 2)

        deci = DefaultDecimation(oversampling=1)
        first_layer = root_node.children[1].children[0].children[0]

        # convolve with stride of 2 (scale 2 wavelet (oversampling))
        # Then convolve again, j2 - j1 is 1 so add a stride of 2
        # to adjust bandwidth to decimated signal
        # Then convolve again  j3 - j2 is 1 so add a stride of 2, decimate
        # wave by 2,
        # we've decimated twice by 2 previously
        w, c = deci.resolve_scales(first_layer)
        self.assertTrue(w == 4 and c == 2)

        deci = DefaultDecimation(oversampling=0)
        first_layer = root_node.children[2].children[0]
        # convolve with stride of 8 (scale 3 wavelet (oversampling))
        # Then convolve again, j2 - j1 is 1 so add a stride of 2
        w, c = deci.resolve_scales(first_layer)
        self.assertTrue(w == 8 and c == 2)
