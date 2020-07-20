import re

import keras.backend as keras_backend
from keras.layers import DepthwiseConv2D

import numpy as np

from traits.api import Float, HasStrictTraits, Instance, Int, Tuple, Property

from blusky.wavelets.i_wavelet_2d import IWavelet2D


class ApplyFatherWavlet2D(HasStrictTraits):
    """
    Provides a "convolution" method that will apply a father wavelet to
    the endpoints of a cascade. Be sure to first apply layers to remove
    any of the padding.

    Assuming the input to the cascade is a power of 2 in shape, the result
    will be a set of scattering coefficients at all orders of the transform
    sampled regularly throughout the image. You can imagine that every
    set of coefficients will be computed at the center of a tile, the shape
    of which is determined by the "J" parameter. The degree to which these
    tiles over lap is controlled by the "overlap_log_2". For interpretation,
    consider values of "J" to give a tile of shape (2**(J+2), 2**(J+2)),
    over which the texture of the image can be considered stationary.

    The tiles can overlap by a factor of "M", however if you use the
    default decimation, you must ensure that you have oversampled enough
    to properly represent the stride at all scales of the transform.
    With default decimation, oversamples=1, overlap_log_2 can be upto
    J - 1. For each unit of overlap, you need to pay the cost of an
    additional unit of oversampling.
    """

    #: (J) This is the "J" scale parameter of the father wavelet used in the
    #  transform.
    J = Int(2)

    #: (M) This is defines the overlap of the tiles, so overlap_log_2 = 0
    #  would be no overlap, overlap_log_2 = 1 would be 50% overlap,
    #  overlap_log_2 = 2 would be 75% etc.
    overlap_log_2 = Int(0)

    #: Size of the image input to the Cascade_2d. This needs to be padded to a
    # power of "2" to ensure that the coefficients are consistent.
    img_size = Tuple

    #: The sample rate of the input data
    sample_rate = Float

    #: Wavelet to use in convolution
    wavelet = Instance(IWavelet2D)

    #: Equivalent tile size derived from the log scale J
    #  J = round(log2(min(tile_size))) - 2
    _tile_size = Property(Int, depends_on="J")

    def _get__tile_size(self):
        size = 2 ** (self.J + 2)
        if size > self.img_size[0] or size > self.img_size[1]:
            mn = min(self.img_size)
            msg = "For image {} by {}, max J is {}".format(
                self.img_size[0], self.img_size[1], np.log2(mn) - 2
            )
            raise RuntimeError(msg)
        return (2 ** (self.J + 2), 2 ** (self.J + 2))

    def _convolve(self, input_layer, trainable=False):
        """
        The concept here is to first derive the applied decimation
        from the shape of the input layer, then pad the layer and
        apply the a convolution with the father wavelet. The padding
        and strideof the convolution is designed to return set of coefficients
        for a collections of regular (optionally overlapping) tiles.
        This will be the case provided the size of the original input to the
        transform are a power of 2.

        Parameters
        ----------
        input_layer - Keras Layer
            A layer to apply the father wavelet to. The applied wavelet
            is derived from the shape of the layer and knowlege of the
            input image shape.

        trainable - Bool (optional)
            Toggle setting the convolution to be trainable. Either way it
            is initialized with a gabor wavelet.

        Returns
        -------
        conv - Keras Layer
            A Keras layer applying the convolution to the input
        """

        # create a convenient name
        name = re.sub("[_/].*", "", input_layer.name)
        name += "phi"

        _, nh, nw, _ = input_layer.shape

        nh = nh
        nw = nw

        # amount of decimation to here.
        factor_1 = self.img_size[0] // nh
        factor_2 = self.img_size[1] // nw

        # how much to decimate the wavelet to required bandwidth
        wavelet_stride = min(factor_1, factor_2)
        # need to guarantee this, ideally crop the wavelet to a
        # power of "2"
        wav = self.wavelet.kernel(
            0.0, shape=(2 ** (self.J + 2) - 1, 2 ** (self.J + 2) - 1)
        )

        #
        wav = wav[::wavelet_stride, ::wavelet_stride]

        # needs to be real
        if np.iscomplexobj(wav):
            wav = wav.real

        # define a little helper to intialize the weights.
        def init_weights(shape, **kwargs):
            dtype = np.float32

            weights = np.zeros(shape, dtype=dtype)

            for ichan in range(shape[2]):
                weights[:, :, ichan, 0] = wav.astype(dtype)

            return keras_backend.variable(value=weights, dtype=dtype)

        # use the father wavelet scale here instead of the default:
        conv_stride = (
            max(
                2 ** (-self.overlap_log_2) * self._tile_size[0] // factor_1, 1
            ),
            max(
                2 ** (-self.overlap_log_2) * self._tile_size[1] // factor_2, 1
            ),
        )
        conv_stride = (int(conv_stride[0]), int(conv_stride[0]))
        
        conv = DepthwiseConv2D(
            name=name,
            kernel_size=wav.shape,
            depth_multiplier=1,
            data_format="channels_last",
            padding="valid",
            strides=conv_stride,
            trainable=trainable,
            depthwise_initializer=init_weights,
        )

        return conv(input_layer)

    def convolve(self, end_points):
        """
        Apply father wavelet convolution.

        Parameters
        ----------
        end_points - List(Keras Layers)
            Typically this would be the multiple end-points of the 2-D Cascade.

        Returns
        -------
        scattering_transform - List(Keras Layers)
            The father wavelet applied to each end-point. The stride and
            padding of the convolution produces a consistent set of
            coefficients at each scale, provided the shape of the original
            image is a power of 2. For example, img.shape = (128, 256).
        """
        scattering_transform = [self._convolve(i) for i in end_points]

        return scattering_transform
