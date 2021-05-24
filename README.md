BluSky - A Python implementation of the wavelet scattering transform
====================================================================

BluSky is a Python library for that implements the Mallat wavelet scattering transform using Keras/Tensorflow.  Features include:
- 1D, 2D transforms.  3D on the way.
- arbitrary order
- Morlets & Gabor, modular library allows for arbitrary wavelets(?)
- built in visualization of transform coefficients

Installation
------------
Install the dependencies in a Conda environment:
```
conda create -n blusky python=3.8.5 traits matplotlib scikit-learn
conda activate blusky
python -m pip install tensorflow==2.4
```

Install BluSky from source:
```
git clone git@github.com:enthought/blusky.git
cd blusky
python -m pip install -e .
```

Getting Started
---------------
- wavelet scattering transform, implemented as a series of convolutions using Keras and Tensorflow.
- following the approach of Bruna, and Mallat (refs)
- the transform is implemented as a convolutional neural network. intermediate activations are easily interrogated to enable easy interrogation of the signal as it propagates through the cascade


Support
-------
This effort was supported by [Sandia National Labs](https://www.sandia.gov/).

![Sandia Labs Logo](/blusky/images/sandia-logo.svg)

with development and maintenance support by [Enthought](https://www.enthought.com).

![Enthought logo](/blusky/images/enthought-logo-vertical.png)
