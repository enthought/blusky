import keras.backend as keras_backend
from keras.layers import DepthwiseConv2D, Lambda, Add
import numpy as np

from traits.api import Enum, HasStrictTraits, Int, Instance, List, Tuple

from blusky.wavelets.i_wavelet_2d import IWavelet2D
from blusky.transforms.cascade_tree import CascadeTree


class Cascade2D(HasStrictTraits):
    """
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

    # provide a list of wavelets to define the cascade, the order is important,
    # the wavelets are applied in order.
    wavelets = List(IWavelet2D)

    # The depth of the transform, how many successive conv/abs iterations
    # to perform, this should be less than or equal to the number of wavelets
    # supplied.
    depth = Int(2)

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

    cascade_tree = Instance(CascadeTree)

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

    def _init_weights(self, shape, dtype=None, wavelet2d=None, real_part=True):
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

        # nx/ny is the image shape, num_inp/outp are the number of
        # channels inpit/output.
        nx, ny, num_inp, num_outp = shape

        if num_outp != len(self.angles):
            raise RuntimeError("weights: mismatch dimension num angles.")

        weights = np.zeros(shape, dtype=dtype)

        for iang, ang in enumerate(self.angles):
            wav = wavelet2d.kernel(ang)

            # keras does 32-bit real number convolutions
            if real_part:
                x = wav.real.astype(np.float32)
            else:
                x = wav.imag.astype(np.float32)

            # we don't want to introduce a phase, put the wavelet
            # in the corner.
            x = np.roll(x, shape[0] // 2, axis=1)
            x = np.roll(x, shape[1] // 2, axis=0)

            # apply to each input channel
            for ichan in range(shape[2]):
                weights[:, :, ichan, iang] = x[: shape[0], : shape[1]]

        return keras_backend.variable(value=weights, dtype=dtype)

    def _convolve_and_abs(
        self, wavelet, inp, stride=1, name="", trainable=False
    ):
        """
        Implement the operations for |inp*psi|. There initially, there
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
        name - Str
            A name for the resulting layer.
        trainable - Bool
            You can think of the scattering transform as a pretrained network,
            so this can be "false" for most applications.
            The "abs" operation is a source of non-linearity, in an edge case
            you might exploit this to train weights for a novel CNN.

        Returns
        -------
        returns - Keras Layer
            The result of the convolution and abs function.

        """
        square = Lambda(lambda x: keras_backend.square(x), trainable=False)
        add = Add(trainable=False)

        # The output gets a special name, because it's here we attach
        # things to.
        sqrt = Lambda(
            lambda x: keras_backend.sqrt(x), trainable=False, name=name
        )
        self._endpoint_counter += 1

        real_part = DepthwiseConv2D(
            kernel_size=wavelet.shape,
            depth_multiplier=len(self.angles),
            data_format="channels_last",
            padding=self._padding,
            strides=stride,
            trainable=trainable,
            depthwise_initializer=lambda args: self._init_weights(
                args, real_part=True, wavelet2d=wavelet
            ),
        )(inp)
        real_part = square(real_part)

        imag_part = DepthwiseConv2D(
            kernel_size=wavelet.shape,
            depth_multiplier=len(self.angles),
            data_format="channels_last",
            padding=self._padding,
            strides=stride,
            trainable=trainable,
            depthwise_initializer=lambda args: self._init_weights(
                args, real_part=False, wavelet2d=wavelet
            ),
        )(inp)
        imag_part = square(imag_part)

        result = add([real_part, imag_part])
        return sqrt(result)

    def _convolve(self, inp, psi, name):
        """
        This computes |inp*psi|.
        Which, for efficiency, (optionally) downsamples the output of the
        convolution.
        """
        # we're not going to downsample on this first pass
        stride = 1  # 2 ** self.stride_log2

        # apply the conv_abs layers
        conv = self._convolve_and_abs(psi, inp, stride=stride, name=name)

        return conv

    def transform(self, inp):
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
        self.cascade_tree = CascadeTree(inp, order=self.depth)
        self.cascade_tree.generate(self.wavelets, self._convolve)
        self.cascade_tree.display()
        return self.cascade_tree.get_convolutions()
