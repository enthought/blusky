import re

from keras.layers import Lambda
from keras.layers.convolutional import ZeroPadding2D

import numpy as np
import tensorflow as tf

from traits.api import Enum, HasStrictTraits, Instance, Int

from blusky.transforms.i_decimation_method import IDecimationMethod


def pad_to_log2(img, mode="reflect", constant_value=0):
    """
    To properly align the transform at multiple scales, we need
    the size of the input image to be a power of 2, so,
    2, 4, 8, 16, ... etc.
    This function applies a padding resize to the nearest (larger)
    power of 2.

    Parameters
    ----------
    img - Array
        A 2d image in numpy format.
    mode - Unicode
        (optional) numpy pad modes,
    constant_value - Float or Int
        (optional) if mode 'constant' use this constant value.

    Returns
    -------
    padded image - Array
        The padded image.
    """

    dh = 2 ** np.ceil(np.log2(img.shape[0])) - img.shape[0]
    dw = 2 ** np.ceil(np.log2(img.shape[1])) - img.shape[1]

    _dh1 = int(dh // 2)
    _dh2 = int(_dh1 + dh % 2)

    _dw1 = int(dw // 2)
    _dw2 = int(_dw1 + dw % 2)

    if mode == "constant":
        return np.pad(
            img,
            ((_dh1, _dh2), (_dw1, _dw2)),
            mode=mode,
            constant_values=constant_value,
        )
    else:
        return np.pad(img, ((_dh1, _dh2), (_dw1, _dw2)), mode=mode)


class ReflectionPadding2D(ZeroPadding2D):
    """ Reimplements Keras' zero-padding layer, adding reflection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        padx, pady = self.padding
        return tf.pad(inputs, ((0, 0), padx, pady, (0, 0)), "REFLECT")


class Pad2D(HasStrictTraits):
    """ Pad on input, then attach yourself to the endpoints of the
        cascade. Padding sizes are estimated from the wavelets to
        be used in the convolutions. Subsequent unpadding requires
        knowledge of how we are decimating.

        Example; re-using the CascadeTree to generate the cascade network, and
        the attach this layer to the endpoints such that the output has the
        padding removed. The job of ensuring the tiles are "unpadded".

        # padding needs to know about how we intend to decimate the
        # convolutions and the size of the wavelets to be used.
        deci = DefaultDecimation(oversampling=1)
        pad_2d = Pad2D(wavelets, decimation=deci)

        # Example, pad the input:
        inp = Input(shape=(nx, ny, nchan))
        padded = pad_2d.pad(inp)

        # assemble convolutions
        cascade_tree = CascadeTree(padded, order=2)
        cascade_tree.generate(wavelets, cascade._convolve)
        convs = cascade_tree.get_convolutions()

        # derive layers for unpadding
        cascade_tree = CascadeTree(padded, order=2)
        cascade_tree.generate(wavelets, pad_2d.unpad)
        unpad = cascade_tree.get_convolutions()

        # constructing a model that pads on input and them unpads:
        unpadded_convs = [i[1](i[0]) for i in zip(convs, unpad)]
        model = Model(inputs=inp, outputs=unpadded_convs)
    """

    #: Conv2d padding method
    conv_padding = Enum("same", "valid")

    #: Decimation method applied a each layer
    decimation = Instance(IDecimationMethod)

    #: the amount to pad in x/y
    _padx = Int(0)
    _pady = Int(0)

    def __init__(self, wavelets, **traits):
        """ conv_padding is used by keras Conv2D, "same" implies that
            keras will pad with zeros where necessary (slow). "valid"
            will not, so the tile will shrink through successive layers,
            and we need to track that to know how to unpad properly.
        """
        super().__init__(**traits)

        # pad to the max dimension of the wavelets
        nx = np.max([wav.shape[0] for wav in wavelets])
        ny = np.max([wav.shape[1] for wav in wavelets])

        # pad by half the size of the wavelet on each end.
        self._padx = 2 ** int(np.ceil(np.log2(nx)) - 1)
        self._pady = 2 ** int(np.ceil(np.log2(ny)) - 1)

    def pad(self, inp):
        """
        Parameters
        ----------
        inp - Keras Layer
            Gets padded.
        wavelets - List(wavelets)
            List of wavelets, looks at these finds the largest and
            uses 1/2 it's size as padding.

        Returns
        -------
        padded - Keras Layer
            Returns padded layer.
        """
        return ReflectionPadding2D((self._padx, self._pady))(inp)

    def unpad(self, inp, wv, node):
        """
        Parameters
        ----------
        inp - Keras Layer
            Gets padded.
        wavelets - List(wavelets)
            List of wavelets, looks at these finds the largest and
            uses 1/2 it's size as padding.
        node - Node
            A node of the cascade tree used to generate the cascade.
            With the same wavelet bank and decimation it generates
            corrects for the successive decimation.

        Returns
        -------
        unpadded - Keras Layer
            Returns a layer to unpadded.
        """
        if self.conv_padding == "same":
            return self._unpad_same(inp, wv, node)
        else:
            return self._unpad_valid(inp, wv, node)

    def _unpad_same(self, inp, wv, node):
        """"""
        name = re.sub("[*,.|_]", "", node.name)
        name += "unpadded"

        #
        wav, conv = self.decimation.resolve_scales(node)

        # the product gives the resulting decimation at
        # the output of this layer, use that to also
        # decimate the padding.
        padx = self._padx // (wav * conv)
        pady = self._pady // (wav * conv)

        return Lambda(
            lambda x: x[:, padx:-padx, pady:-pady, :],
            trainable=False,
            name=name,
        )

    def _unpad_valid(self, inp, wv, node):
        raise NotImplementedError()
