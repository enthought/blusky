import numpy as np

from traits.api import (
    Bool,
    Float,
    HasStrictTraits,
    Int,
    Property,
    provides,
    Tuple,
)

from blusky.wavelets.i_wavelet_1d import IWavelet1D



@provides(IWavelet1D)
class Morlet1D(HasStrictTraits):
    """
    """

    #: If the wavelet with eccentricity, the orientation of
    # this defines the orientation of its principle axis.
    orientation = Float

    #: bandwidth measured as full-width a half maximum (-3db) of the
    # gaussian envelope in the frequency domain.
    # The two numbers define fwhm in orthogonal directions.
    bandwidth = Tuple(Float)

    #: The center frequency along the principle axis
    center_frequency = Float

    #: sample rate in units
    sample_rate = Float

    #: Where (in std deviations) of the envelope in the spatial domain,
    # to crop the wavelet. Should be >3 (generally) smaller values
    # will be more efficient to convolve, but have less fidelity in
    # the frequency domain.
    crop = Float(6.0)

    #: Optionally taper the cropped image.
    taper = Bool(False)

    #: To build a convolutional model, trade-off fidelity with
    # computation cost (small the better).
    shape = Tuple(Int)

    #: (Optional) labels scale of wavelet, makes sense in a filter bank.
    scale = Int(-1)

    #: measured in "samples"
    _sigma = Property(
        Tuple(Float),
        depends_on=["bandwidth", "center_frequency", "sample_rate"],
    )

    def __init__(self, center_frequency, bandwidth, sample_rate, **traits):
        """
        Parameters
        ----------
        center_frequency - Float
           Specify the center frequency (cycles/units) measured along the
           minor axis of the wavelet.

        bandwidth - Float
           Specify the bandwidth in the major/minor axes (cycles/units)
           of the wavelet.

        sample_rate - Float
           Specify units/sample.

        Optional keyword argument:

        crop - Float
           Specifies a multiple of the envelope to crop the image for output.

        taper - Bool
           If true, applies a hanning window to the image on output. This
           maybe useful for reducing edge effects in subsequent convolutions.

        usage:

        wav = Morlet1D(sample_rate=0.004,
                       center_frequency=60.,
                       bandwidth=30.)
        """

        self.center_frequency = center_frequency
        self.bandwidth = bandwidth
        self.sample_rate = sample_rate

        super().__init__(**traits)

    def _get__sigma(self):
        """
        sigma parameterizes the gaussian envelope of the wavelet.
        Measure the bandwidth at the FWHM, the fouier spectrum is
        gaussian with standard deviation: sigma' = 1/sigma
        bandwidth, measured at FWHM ~ 2.355 / sigma
        """

        def to_ang(f):
            return 2 * np.pi * f * self.sample_rate

        return tuple([2.355 / to_ang(f) for f in self.bandwidth])

    def _shape_default(self):
        """
        Define a square large enough to hold the wavelet in
        any orientation.
        """
        # tiles are square
        _n = np.int_(self.crop * max(self._sigma))
        
        return (_n,)

    def _taper(self):
        """ Compute hanning window to taper image.
        """
        taper = np.kaiser(self.shape[0], 3)
        return taper

    def kernel(self, shape=None):
        """
        Output the wavelet in an complex valued array.

        Derivative of the work: morlet_2d_pyramid.m, we applied the
        same idea to 1-d for consistency.

        from https://github.com/scatnet/scatnet

        See license in NOTICE.txt in this directory.

        Parameters
        ----------
        length - Int (optional)
           Provide a required length for the output wavelet. If not provided
           it will use defaults determined by crop.

        Return
        ------
        wavelet - Array
           A 2d array containing the wavelet
        """

        if shape is None:
            N = self.shape[0]
        else:
            N = shape

        x = np.arange(N)
        x -= N // 2

        # convert to units of cycles per sample
        xi = 2 * np.pi * self.center_frequency * self.sample_rate

        gaussian_envelope = np.exp(-x * x / (2 * (self._sigma[0] ** 2)))
        oscilating_part = gaussian_envelope * np.exp(1j * x * xi)
        K = np.sum(oscilating_part) / np.sum(gaussian_envelope)
        gabc = oscilating_part - K * gaussian_envelope

        normalized_wavelet = gabc / (np.abs(gabc).sum())

        if self.taper:
            normalized_wavelet *= self._taper()

        # 1.3333 aligns the amplitude with scatnet impl.
        return normalized_wavelet * 1.3333
