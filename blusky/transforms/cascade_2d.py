import re

import keras.backend as keras_backend
from keras.layers import DepthwiseConv2D, Lambda, Add
import numpy as np

from traits.api import Enum, HasStrictTraits, Int, Instance, Tuple

from blusky.transforms.default_decimation import NoDecimation
from blusky.transforms.i_decimation_method import IDecimationMethod


class Cascade2D(HasStrictTraits):
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

    # Stride - Set a stride when applying the convolutions:
    # Interpreted as "factor 2^n", to apply at each convolution.
    # each step, "0" gives a stride of 1 sample (the default).
    # 1 will apply convolutions at every second sample.
    # For now, negative numbers are not valid.
    stride_log2 = Int(0)

    # Size of the poolin to apply at each step, "0" means no pooling,
    # negative numbers will cause the output to be upsampled by that factor
    pooling_size = Int

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

    def __init__(self, pooling_type, stride_log2, **traits):
        self.pooling_type = pooling_type.lower()
        if stride_log2 < 0:
            raise RuntimeError(
                "stride_log2 needs to be > 0, we don't support \
                upsampling right now."
            )

        super().__init__(**traits)

    def _init_weights(
        self, shape, node=None, dtype=None, wavelet2d=None, real_part=True
    ):
        """
        Create an initializer for DepthwiseConv2D layers. We need these
        layers instead of Conv2D because we don't want it to stack across
        channels.

        Parameters
        ----------
        wavelet2d - IWavelet2D
            An object to create a wavelet.

        dtype - Float
            Data type for the wavelet, default is float32

        real_part - Bool
            If true it will initialize the convolutional weights
            with the real-part of the wavelet, if false, the
            imaginary part.

        Returns
        -------
        returns - tensorflow variable
            returns a tensorflow variable containing the weights.

        """
        if dtype is None:
            dtype = np.float32

        # precompute decimation
        wavelet_stride, conv_stride = self.decimation.resolve_scales(node)

        # we need to normalize by the decimation factor to preserve amplitude
        deci_norm = (wavelet_stride * conv_stride)**2

        # nx/ny is the image shape, num_inp/outp are the number of
        # channels inpit/output.
        nx, ny, num_inp, num_outp = shape

        if num_outp != len(self.angles):
            raise RuntimeError("weights: mismatch dimension num angles.")

        weights = np.zeros(shape, dtype=dtype)

        for iang, ang in enumerate(self.angles):
            wav = wavelet2d.kernel(ang) * deci_norm

            # decimate wavelet
            wav = self.decimation.decimate_wavelet(wav, wavelet_stride)

            # keras does 32-bit real number convolutions
            if real_part:
                x = wav.real.astype(np.float32)
            else:
                x = wav.imag.astype(np.float32)

            # apply to each input channel
            for ichan in range(shape[2]):
                weights[:, :, ichan, iang] = x[: shape[0], : shape[1]]

        return keras_backend.variable(value=weights, dtype=dtype)

    def _convolve_and_abs(self, wavelet, inp, node, trainable=False):
        """
        Implement the operations for |inp*psi|. Initially, there
        will be a channel for each angle defined in the cascade. For
        subsequent convolutions, a abs/conv operation is applied to
        each channel in the input, for each angle defined in the
        cascade.

        For example, if we have 3-angles defined in the cascade (angles)

        transform order | number of channels output
        ----------------------
        order 1         | 3-channels
        order 2         | 9-channels
        order 3         | 27-channels


        Parameters
        ----------
        wavelet - IWavelet2D
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
        # create a valid layer name
        name = re.sub("[*,.|_]", "", node.name)

        #
        wavelet_stride, conv_stride = self.decimation.resolve_scales(node)

        # after decimation
        wavelet_shape = (
            wavelet.shape[0] // wavelet_stride,
            wavelet.shape[1] // wavelet_stride,
        )

        square = Lambda(lambda x: keras_backend.square(x), trainable=False)
        add = Add(trainable=False)

        # The output gets a special name, because it's here we attach
        # things to. We name to the (endpoint)
        sqrt = Lambda(
            lambda x: keras_backend.sqrt(x), trainable=False, name=name
        )
        self._endpoint_counter += 1

        def real_init(*args, **kwargs):
            return self._init_weights(*args,
                                      node=node,
                                      real_part=True,
                                      wavelet2d=wavelet)

        def imag_init(*args, **kwargs):
            return self._init_weights(*args,
                                      node=node,
                                      real_part=False,
                                      wavelet2d=wavelet)

        
        real_part = DepthwiseConv2D(
            kernel_size=wavelet_shape,
            depth_multiplier=len(self.angles),
            data_format="channels_last",
            padding=self._padding,
            strides=conv_stride,
            trainable=trainable,
            depthwise_initializer=real_init,
        )(inp)
        real_part = square(real_part)

        imag_part = DepthwiseConv2D(
            kernel_size=wavelet_shape,
            depth_multiplier=len(self.angles),
            data_format="channels_last",
            padding=self._padding,
            strides=conv_stride,
            trainable=trainable,
            depthwise_initializer=imag_init,
        )(inp)
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
