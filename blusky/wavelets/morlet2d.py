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

from blusky.wavelets.i_wavelet_2d import IWavelet2D


@provides(IWavelet2D)
class Morlet2D(HasStrictTraits):
    """
    Construct a 2-D Morlet wavelet using parameters of center frequency,
    bandwidth and sample rate. Defining a difference in bandwidth will
    result in an eccentricity of the wavelet.

    The kernel method takes a "theta" parameter to be used to rotate
    the wavelet in 2D. The shape of the output 2d wavelet

    Optional keyword argument:

    crop - Float
       Specifies a multiple of the envelope to crop the image for output.

    taper - Bool
       If true, applies a hanning window to the image on output. This maybe
       useful for reducing edge effects in subsequent convolutions.

    """

    #: If the wavelet with eccentricity, the orientation of
    # this defines the orientation of its principle axis.
    orientation = Float

    #: bandwidth measured as full-width a half maximum (-3db) of the
    # gaussian envelope in the frequency domain.
    # The two numbers define fwhm in orthogonal directions.
    bandwidth = Tuple(Float, Float)

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
    shape = Property(Tuple(Int, Int), depends_on=["_sigma"])

    #: (Optional) labels scale of wavelet, makes osense in a filter bank.
    # Defaults to -1, which is an invalid scale number.  If the scale
    # is not explicitly set, then scale will be inferred from ordering.
    scale = Int(-1)

    #: measured in "samples"
    _sigma = Property(
        Tuple(Float, Float),
        depends_on=["bandwidth", "center_frequency", "sample_rate"],
    )

    def __init__(self, center_frequency, bandwidth, sample_rate, **traits):
        """
        Parameters
        ----------
        center_frequency - Float
           Specify the center frequency (cycles/units) measured along the
           minor axis of the wavelet.

        bandwidth - Tuple(Float, Float)
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

        wav = Morlet2D(sample_rate=0.004,
                       center_frequency=60.,
                       bandwidth=(30.,15.))
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

    def _get_shape(self):
        """
        Define a square large enough to hold the wavelet in
        any orientation.
        """
        # tiles are square
        _n = np.int_(self.crop * max(self._sigma))
        # nicer if even
        _n += 1 - (_n % 2)
        return (_n, _n)

    def _taper(self):
        """ Compute hanning window to taper image.
        """
        taper = np.outer(
            np.kaiser(self.shape[0], 3), np.kaiser(self.shape[1], 3)
        )
        return taper

    def kernel(self, theta, shape=None):
        """
        Output the wavelet in an complex valued array.

        Derivative of the work: morlet_2d_pyramid.m

        from https://github.com/scatnet/scatnet

        See license in NOTICE.txt in this directory.

        Parameters
        ----------
        theta - Float
           The orientation of the major axis (in degrees).

        Return
        ------
        wavelet - Array
           A 2d array containing the wavelet
        """

        # convert to radians
        _theta = np.deg2rad(theta)

        if shape is None:
            N, M = self.shape
        else:
            N, M = shape

        #
        X, Y = np.meshgrid(np.arange(M), np.arange(N))

        # the gaussian envelope is measured in samples
        X = X - M // 2
        Y = Y - M // 2

        # rotation matrix
        Rth = np.array(
            [
                [np.cos(_theta), np.sin(_theta)],
                [-np.sin(_theta), np.cos(_theta)],
            ]
        )

        # envelope in the major/minor directions
        vec = np.array(
            [[1 / self._sigma[0] ** 2.0, 0], [0, 1.0 / self._sigma[1] ** 2.0]]
        )

        # Orientate the major axis
        A = Rth.T.dot(vec).dot(Rth)

        # element by element
        S = X * (A[0, 0] * X + A[0, 1] * Y) + Y * (A[1, 0] * X + A[1, 1] * Y)

        # Gaussian envelope in spatial domain
        gaussian_envelope = np.exp(-S / 2)

        # convert to units of cycles per sample
        xi = 2 * np.pi * self.center_frequency * self.sample_rate

        # (Proof by taking fourier transform), the complex phase
        # shifts the spectrum to be distributed about the center
        # frequency. The variance is the inverse of sigma^2.
        oscilating_part = gaussian_envelope * np.exp(
            1j * (X * xi * np.cos(_theta) + Y * xi * np.sin(_theta))
        )

        K = np.sum(oscilating_part) / np.sum(gaussian_envelope)

        #
        gabc = oscilating_part - K * gaussian_envelope

        #
        normalized_wavelet = gabc / (
            2 * np.pi * self._sigma[0] * self._sigma[1]
        )

        if self.taper:
            normalized_wavelet *= self._taper()

        return normalized_wavelet
