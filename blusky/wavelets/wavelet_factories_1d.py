import numpy as np

from blusky.wavelets.gabor_1d import Gabor1D
from blusky.wavelets.morlet_1d import Morlet1D


def calibrate_wavelets_1d(wavelets):
    """calibrate wavelet shapes to align with valid padding"""
    ni = np.argmin([wav.shape[0] for wav in wavelets])
    
    _shape = wavelets[ni].shape[0]
    smallest_wav = int(2**np.ceil(np.log2(_shape)))    
    smallest_j = wavelets[ni].scale
    
    for wav in wavelets:
        wav.shape = (2**(wav.scale - smallest_j)*smallest_wav,)
        
def morlet_freq_1d(J):
    """ derived from scatnet/filters/morlet1d see NOTICE.txt
        for license declaration. """
    
    sigma0 = 2/np.sqrt(3);

    Q = 1.
    B = 1.
    
    _xi_psi = 1/2*(2**(-1.)+1)*np.pi
    _sigma_psi = 1/2*sigma0/(1-2**(-1/B))
    
    phi_bw_multiplier = 1+(Q==1);
    _sigma_phi = _sigma_psi/phi_bw_multiplier
        
    P = int(round((2.**(-1/Q)- 1/4*sigma0/_sigma_phi )/(1-2.**(-1/Q))))
    
    # Calculate logarithmically spaced, band-pass filters.
    #xi_psi = xi_psi * 2.^((0:-1:1-J)/Q);
    xi_psi = _xi_psi * 2.**(-np.arange(0,J)/Q)
    
    #sigma_psi = sigma_psi * 2.^((0:filt_opt.J-1)/filt_opt.Q);
    sigma_psi = _sigma_psi * 2**(np.arange(0,J)/Q)
    
    # Calculate linearly spaced band-pass filters so that they evenly
    # cover the remaining part of the spectrum
    #step = pi * 2^(-filt_opt.J/filt_opt.Q) * ...
    #    (1-1/4*sigma0/filt_opt.sigma_phi*2^(1/filt_opt.Q))/filt_opt.P;
    step = np.pi * 2**(-J/Q) * (1-1/4*sigma0/_sigma_phi*2**(1./Q))/P
        
    # remove "J"+1, added "P"+1 for Python 
    xi_psi[J:J+P+1] = _xi_psi * 2**((-J+1)/Q) - step * np.arange(1,P+1)
     
    # remove "J"+1, added "P"+1 for Python 
    sigma_psi[J:J+1+P+1] = _sigma_psi*2**((J-1)/Q)
    
    # Calculate low-pass filter
    sigma_phi = _sigma_phi * 2.**((J-1.)/Q);

    return xi_psi, sigma_psi, sigma_phi


def vanilla_morlet_1d(
        sample_rate, J, j=0, sigma=None, xi=None, crop=5, taper=False):
    """ """

    if j < 0:
        raise RuntimeError("Resolution parameter j is strictly positive")

    if j >= J:
        raise RuntimeError("""Resolution parameter j must be less than 
                            log-scale J""")
    
    xi_psi, sigma_psi, _ = morlet_freq_1d(J)
    if xi is None:
        xi = xi_psi[j]*(2**j)
    if sigma is None:
        sigma = sigma_psi[j]/(2**j)
        
    _bandwidth = 2.355 / (2 * np.pi * sigma * sample_rate)
    center_frequency = xi / (2 * np.pi * sample_rate)

    # rescale for changes in resolution:
    _bandwidth /= 2**j
    center_frequency /= 2**j

    bandwidth = (_bandwidth,)

    # Equivalent Morlet wavelet
    wav = Morlet1D(
        sample_rate=sample_rate,
        center_frequency=center_frequency,
        bandwidth=bandwidth,
        crop=crop,
        taper=taper,
        scale=j+1,  # defines the rate at which we decimate
    )

    return wav


def vanilla_gabor_1d(
        sample_rate, J, sigma=None, xi=None, crop=5, taper=False
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

    # provides defaults
    _, _, sigma_phi = morlet_freq_1d(J)
    if xi is None:
        xi = 0.0
    if sigma is None:
        sigma = sigma_phi/(2**J)
    
    _bandwidth = 2.355 / (2 * np.pi * sigma * sample_rate)
    center_frequency = xi / (2 * np.pi * sample_rate)

    # rescale for changes in resolution:
    _bandwidth /= 2 ** J
    center_frequency /= 2 ** J

    bandwidth = (_bandwidth,)

    # Equivalent Gabor wavelet
    wav = Gabor1D(
        sample_rate=sample_rate,
        center_frequency=center_frequency,
        bandwidth=bandwidth,
        crop=crop,
        taper=taper,
        scale=J,  # defines the rate at which we decimate
    )

    return wav
