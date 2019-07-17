
import numpy as np

from blusky.wavelets.morlet2d import Morlet2D

def scale_morlet_2d(morlet_2d, j):
    """
    Rescale the input wavelet according to "j".

    Parameters
    ----------
    morlet_2d - Morlet2D
        Reference wavelet (scale j=0)
    j - Int
        Scale parameter, j>0.
    
    Returns
    -------
    wav - Morlet2D
        Rescaled Morlet wavelet.
    """

    if j < 0:
        raise RuntimeError("Resolution parameter j is strictly positive")
    
    _bandwidth = morlet_2d.bandwidth
    _center_frequency = morlet_2d.bandwidth
    
    # rescale for changes in resolution:
    _bandwidth = (bandwidth[0] * 2**j, bandwidth[1] * 2**j)
    _center_frequency = center_frequency/2**j

    # Equivalent Morlet wavelet
    wav = Morlet2D(sample_rate=morlet_2d.sample_rate, 
                   center_frequency=_center_frequency, 
                   bandwidth=_bandwidth, 
                   crop=morlet_2d.crop, 
                   taper=morlet_2d.taper)
    
    return wav

def vanilla_morlet_2d(sample_rate, j=0, sigma=0.8, slant=0.5, xi=2.35619,
                     crop=5, taper=False):
    """
    This is a lexicon that translates to our definition of the wavelet in 
    physical units to the definitions of https://github.com/scatnet/scatnet
    that works with dimensionless units. Returns a Morlet2D wavelet 
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

    slant - Float
        controls the relative bandwidth

    crop - Float
       Specifies a multiple of the envelope to crop the image for output.

    taper - Bool
       If true, applies a hanning window to the image on output. This maybe
       useful for reducing edge effects in subsequent convolutions.
    
    Returns
    -------
    wav - Morlet2D
        2D Morlet wavelet constructed according the dimensionless definitions.
    """
    
    if j < 0:
        raise RuntimeError("Resolution parameter j is strictly positive")
    
    _bandwidth = 2.355 / (2 * np.pi * sigma * sample_rate)
    center_frequency = xi / (2 * np.pi * sample_rate )
    
    # rescale for changes in resolution:
    _bandwidth /= 2**j
    center_frequency /= 2**j
    
    bandwidth = (_bandwidth, _bandwidth*slant)
    
    # Equivalent Morlet wavelet
    wav = Morlet2D(sample_rate=sample_rate, 
                   center_frequency=center_frequency, 
                   bandwidth=bandwidth, 
                   crop=crop, 
                   taper=taper)
    
    return wav
