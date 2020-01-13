import numpy as np

from traits.api import HasStrictTraits, Int, provides

from blusky.transforms.i_decimation_method import IDecimationMethod


@provides(IDecimationMethod)
class DefaultDecimation(HasStrictTraits):
    """
    By default we'll decimate the wavelet by simple decimation,
    and decimate the convolution by setting a stride.

    An issue with this implementation is that by simple decimation,
    at the later order; a wavelet of scale "j" decimated "j" times will
    show aliasing.

    A remedy for this is to oversample to transform by a factor of "2";
    essentially doubling the overall cost of the compute.
    """

    #: The oversampling factor of 2, "2"-should be enough to avoid aliasing.
    #  values of 0, 1, 2 are valid.
    oversampling = Int(1)

    def resolve_scales(self, node):
        """
        Recurse through the cascade tree and compute factors
        to decimate, both the wavelet and transform. Apply the oversample
        factor

        Parameters
        ----------
        cascade_tree - Node
            A node in a tree structure that tracks the scale and order of the
            transform.

        Returns
        -------
        wavelet_factor - Int
            The factor to decimate the wavelet.
        conv_factor - Int
            The factor to decimate the resulting convolution.
        """

        if node.parent is None:
            return 1, 1

        scales = []
        _node = node
        while _node.parent is not None:
            _scale = max(_node.scale - self.oversampling, 0)
            _parent_scale = max(_node.parent.scale - self.oversampling, 0)

            scales.append(_scale - _parent_scale)

            _node = _node.parent

        # cumulative decimation by scale along scattering path
        cum_deci = np.cumsum(scales[::-1])[::-1]

        # how much to decimate to get to here
        if len(cum_deci) > 1:
            # total factors of 2 decimated to upto this layer
            wavelet_factor = cum_deci[1]
        else:
            wavelet_factor = 0

        # the stride at this layer
        conv_factor = scales[0]

        return 2 ** wavelet_factor, 2 ** conv_factor

    def decimate_wavelet(self, wav, factor):
        """ Apply decimation to wavelet.

        Returns
        -------
        wavelet - Array
        """
        if factor < 2:
            return wav
        else:
            # applies stride to each dimension 1d, 2d, ...
            return wav[::factor]

    def decimate_convolution(self, inp, factor):
        msg = "Just use the conv_factor as the stride in the Conv layer."
        raise NotImplementedError(msg)


@provides(IDecimationMethod)
class NoDecimation(HasStrictTraits):
    """
    Inefficient, but mathematically correct. Use this for testing.
    """

    def resolve_scales(self, node):
        """
        Doesn't matter, always returns "1".

        Parameters
        ----------
        cascade_tree - Node
            A node in a tree structure that tracks the scale and order of the
            transform.

        Returns
        -------
        wavelet_factor - Int
            The factor to decimate the wavelet.
        conv_factor - Int
            The factor to decimate the resulting convolution.
        """
        return 1, 1

    def decimate_wavelet(self, wav, factor):
        """ Doesn't apply decimation to wavelet.
        Returns
        -------
        wavelet - Array
        """
        return wav

    def decimate_convolution(self, inp, factor):
        """ Doesn't do anything.
        """
        return inp
