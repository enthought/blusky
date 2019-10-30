import numpy as np

from blusky.wavelets.gabor_1d import Gabor1D
from blusky.wavelets.morlet_1d import Morlet1D


def vanilla_morlet_1d(
    sample_rate, j=0, sigma=1.15, xi=2.35619, crop=5, taper=False
):
    """
    This is a lexicon that translates to our definition of the wavelet in
    physical units to the definitions of https://github.com/scatnet/scatnet
    that works with dimensionless units. Returns a Morlet1D wavelet
    constructed according the dimensionless definitions.

    Find bandwidth to get required (sigma):

    sigma' = 2.355 / (2 * np.pi * bandwidth * sample_rate) (physical units)
          = sigma (dimensionless units)

    required bandwidth:

    bandwidth = 2.355 / (2 * np.pi * sigma * sample_rate)

    Find center_frequency to get required (xi):

    xi' = 2 * np.pi * center_frequency * sample_rate (physical units)
        = xi (dimensionless units)

    required center_frequency:

    center_frequency = xi / (2 * np.pi * sample_rate )

    Changing resolution:

    x -> x' = 2**-j x

    Equivalent to:

    sigma -> sigma' * 2**j
    xi -> xi' = xi/2**j

    Parameters
    ----------
    sample_rate - Float
        Specify units/sample

    j - Int
        The scale of the wavelet (0,1,2, ....)

    sigma - Float
        standard deviation controlling bandwidth in dimensionless units

    xi - Float
        center frequency in dimensionless units

    crop - Float
       Specifies a multiple of the envelope to crop the image for output.

    taper - Bool
       If true, applies a hanning window to the image on output. This maybe
       useful for reducing edge effects in subsequent convolutions.

    Returns
    -------
    wav - Morlet1D
        1D Morlet wavelet constructed according the dimensionless definitions.
    """

    if j < 0:
        raise RuntimeError("Resolution parameter j is strictly positive")

    _bandwidth = 2.355 / (2 * np.pi * sigma * sample_rate)
    center_frequency = xi / (2 * np.pi * sample_rate)

    # rescale for changes in resolution:
    _bandwidth /= 2 ** j
    center_frequency /= 2 ** j

    bandwidth = (_bandwidth,)
    
    # Equivalent Morlet wavelet
    wav = Morlet1D(
        sample_rate=sample_rate,
        center_frequency=center_frequency,
        bandwidth=bandwidth,
        crop=crop,
        taper=taper,
        scale=j + 1,  # defines the rate at which we decimate
    )

    return wav


def vanilla_gabor_1d(
    sample_rate, j=0, sigma=1.15, xi=0.0, crop=5, taper=False
):
    """
    This is a lexicon that translates to our definition of the wavelet in
    physical units to the definitions of https://github.com/scatnet/scatnet
    that works with dimensionless units. Returns a Gabor2D wavelet
    constructed according the dimensionless definitions.

    Find bandwidth to get required (sigma):

    sigma' = 2.355 / (2 * np.pi * bandwidth * sample_rate) (physical units)
          = sigma (dimensionless units)

    required bandwidth:

    bandwidth = 2.355 / (2 * np.pi * sigma * sample_rate)

    Find center_frequency to get required (xi):

    xi' = 2 * np.pi * center_frequency * sample_rate (physical units)
        = xi (dimensionless units)

    required center_frequency:

    center_frequency = xi / (2 * np.pi * sample_rate )

    Changing resolution:

    x -> x' = 2**-j x

    Equivalent to:

    sigma -> sigma' * 2**j
    xi -> xi' = xi/2**j

    Parameters
    ----------
    sample_rate - Float
        Specify units/sample
    j - Int
        The scale of the wavelet (0,1,2, ....)

    sigma - Float
        standard deviation controlling bandwidth in dimensionless units

    xi - Float
        center frequency in dimensionless units

    crop - Float
       Specifies a multiple of the envelope to crop the image for output.

    taper - Bool
       If true, applies a hanning window to the image on output. This maybe
       useful for reducing edge effects in subsequent convolutions.

    Returns
    -------
    wav - Gabor1D
        1D Gabor wavelet constructed according the dimensionless definitions.
    """

    if j < 0:
        raise RuntimeError("Resolution parameter j is strictly positive")

    _bandwidth = 2.355 / (2 * np.pi * sigma * sample_rate)
    center_frequency = xi / (2 * np.pi * sample_rate)

    # rescale for changes in resolution:
    _bandwidth /= 2 ** j
    center_frequency /= 2 ** j

    bandwidth = (_bandwidth,)

    # Equivalent Gabor wavelet
    wav = Gabor1D(
        sample_rate=sample_rate,
        center_frequency=center_frequency,
        bandwidth=bandwidth,
        crop=crop,
        taper=taper,
        scale=j + 1,  # defines the rate at which we decimate
    )

    return wav
