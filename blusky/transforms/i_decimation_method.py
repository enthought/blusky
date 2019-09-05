from traits.api import Interface


class IDecimationMethod(Interface):
    """ Define objects to control how to decimate at sucessive
    iterations of the cascade.
    """

    def resolve_scales(self, node):
        """ Recurse through the cascade tree and compute factors
        to decimate, both the wavelet and transform.

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
        raise NotImplementedError()

    def decimate_wavelet(self):
        """ Apply decimation to wavelet.
        Returns
        -------
        wavelet - Array
        """
        raise NotImplementedError()

    def decimate_convolution(self, inp, factor):
        """ Apply decimation to the convolution.
        Parameters
        ----------
        inp - Keras Layer
            Layer to decimate.
        factor - Int
            Amount to decimate by.
        Returns
        -------
        conv_layer - Keras Layer
        """
        raise NotImplementedError()
