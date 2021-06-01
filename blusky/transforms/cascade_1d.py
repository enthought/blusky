import numpy as np
import re
import sys

import tensorflow.keras.backend as keras_backend
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Conv1D, Lambda, Add, SeparableConv1D
from tensorflow import print as tf_print

from traits.api import Enum, HasStrictTraits, Int, Instance, Tuple

from blusky.transforms.default_decimation import NoDecimation
from blusky.transforms.i_decimation_method import IDecimationMethod
from blusky.transforms.blusky_net_weights_1d import BluskyNetWeights1D
from blusky.utils.pad_1d import ReflectionPadding1D


class Cascade1D(HasStrictTraits):
    """
    Caution this has a bug.

    The idea here is to implement a cascade of convolvolution
    and modulus opertations.
    Suppose I had a sequence of wavelets, psi1, psi2, ...

    |x * psi1|
    |x * psi2| -> output
        .
        .
        .
     ---> ||x * psi1| * psi2|
          ||x * psi1| * psi3|
                .               -> output
                .
          ||x * psi2| * psi3|
                .
                .
                  |
                  |
                  ---> .. etc ..
    """

    #: Provides methods for decimating at each layer in the transform.
    decimation = Instance(IDecimationMethod, NoDecimation())

    #: Subsequent convolutions can be applied to downsampled images for
    #  efficiency.
    # Provide some options with Keras for this:
    # Max - MaxPooling (take max value in a window)
    # Average - AveragePooling (average values in window)
    pooling_type = Enum(["none", "max", "average"])

    # shape of the input tile
    shape = Tuple(Int)

    #: In 2D we will apply the transform over a set of wavelets are different
    # orientation, define that here in degrees.
    angles = Tuple

    #: Direction to Keras Conv2d on how to do padding at each convolution,
    #  "same" pads with zeros, "valid" doesn't. This doesn't replace the
    #  need to pad tiles during preprocessing, however, "same" will maintain
    #  tile size through each layer.
    #  "valid" is faster.
    _padding = Enum(["same", "valid"])

    #: private, labels endpoints to attach things to
    _endpoint_counter = Int(0)

    #:
    _current_order = Int(1)

    def _convolve_and_abs(self, wavelet, inp, node, trainable=False):
        """
        Implement the operations for |inp*psi| in 1-D. Assumes a single
        input channel. If you have multiple input channels you want
        the convolution to apply to, then to each independently.

        Parameters
        ----------
        wavelet - IWavelet1D
            A wavelet object used to generate weights for each angles,
            defined in self.angles.
        inp - Keras Layer
            A keras layer to apply the convolution to. For example,
            an input layer. Or subsequently the output of the previous
            convolutions.
        stride - Int
            Set a stride across the convolutions. This should be determined
            by the scale of the transform.
        node - Node
            Node in the tree.

        Returns
        -------
        returns - Keras Layer
            The result of the convolution and abs function.
        """

        # Sample/decimate the wavelet
        wav = wavelet.kernel()

        # precompute decimation
        wavelet_stride, conv_stride = self.decimation.resolve_scales(node)

        # we need to normalize by the decimation factor to preserve amplitude
        deci_norm = (wavelet_stride * conv_stride)

        # decimate wavelet
        wav = self.decimation.decimate_wavelet(wav, wavelet_stride)
        wav *= deci_norm
        
        wavelet_shape = wav.shape

        # create a valid layer name
        name = re.sub("[*,.|_]", "", node.name)

        square = Lambda(lambda x: keras_backend.square(x), trainable=False)
        
        add = Add(trainable=False)

        # The output gets a special name, because it's here we attach
        # things to. We name to the (endpoint)
        sqrt = Lambda(
            lambda x: keras_backend.sqrt(x), trainable=False, name=name
        )
        self._endpoint_counter += 1

        # ensures proper alignment of subsequent convolutions
        if self._padding == "valid":
            _valid_align = int(wavelet_shape[0]//2)
            inp = ReflectionPadding1D((_valid_align,
                                       _valid_align-1))(inp)
            
        real_part = Conv1D(
            1,
            kernel_size=wavelet_shape,
            data_format="channels_last",
            padding=self._padding,
            strides=conv_stride,
            trainable=trainable,
            use_bias=False,
            kernel_initializer=BluskyNetWeights1D(wav.real))(inp)

        real_part = square(real_part)

        imag_part = Conv1D(
            1,
            kernel_size=wavelet_shape,
            data_format="channels_last",
            padding=self._padding,
            strides=conv_stride,
            trainable=trainable,
            use_bias=False,
            kernel_initializer=BluskyNetWeights1D(wav.imag))(inp)
        
        imag_part = square(imag_part)

        result = add([real_part, imag_part])
        
        return sqrt(result)

    def _convolve(self, inp, psi, node):
        """
        This computes |inp*psi|.
        Which, for efficiency, (optionally) downsamples the output of the
        convolution.
        """
        # apply the conv_abs layers
        conv = self._convolve_and_abs(psi, inp, node)

        return conv

    def transform(self, cascade_tree, wavelets):
        """
        Apply abs/conv operations to arbitrary order.
        Doesn't apply the DC term, just the subsequent layers.

        Parameters
        ----------
        inp - Keras Layer
            The input at the root of the cascade. Would generally
            be a Keras Input Layer.

        Returns
        -------
        returns - Keras Model
            Returns a keras model applying the conv/abs operations
            of the scattering transform to the input.
        """
        cascade_tree.generate(wavelets, self._convolve)

        return cascade_tree.get_convolutions()
