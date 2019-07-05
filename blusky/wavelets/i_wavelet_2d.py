from traits.api import Interface


class IWavelet2D(Interface):
    """
    Provides an object model for a 2d wavelet.

    For consistency we expect these to be parameterized with physical values
    such as center frequency, bandwidth and sample rate. The wavelet
    object should be able to determine the best shape for its buffer,
    with some flexibility for the user to change this. It should also,
    optionally, taper its output.
    """

    def kernel(self, *args):
        """
        Output the wavelet in an complex valued array.
        """
        raise NotImplementedError()
