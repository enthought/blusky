# BluSky - A Python implementation of the wavelet scattering transform

BluSky is a Python library for that implements the Mallat wavelet scattering transform using Keras/Tensorflow.  Features include:
- 1D, 2D transforms.  3D on the way.
- arbitrary order
- Morlets & Gabor, modular library allows for arbitrary wavelets(?)
- built in visualization of transform coefficients

## Installation
---

### Dependencies

The requirements for installing BluSky are:

* numpy
* traits 
* pillow
* tensorflow<=1.14
* keras<=2.2.4-8
* matplotlib

### CPU Installation
Instructions to install from scratch.  Show edm instructions, as well as conda.

`edm install tensorflow keras`
TODO: test this install proceure

Should include conda install instructions (eventually).

Does anything need to be `pip install`'ed after `edm`?

### GPU Installation

Include instructions for getting this to run on a GPU.

`edm install tensorflow-gpu`...

### Install from source

## Getting Started
---
- wavelet scattering transform, implemented as a series of convolutions using Keras and Tensorflow.
- following the approach of Bruna, and Mallat (refs)
- the transform is implemented as a convolutional neural network. intermediate activations are easily interrogated to enable easy interrogation of the signal as it propagates through the cascade

This [introductory notebook](https://www.github.com/notebooks/BluSky%20-%20Getting%20Started.ipynb) describes the basic operation of the library, how to create the transform object and apply it to sample data.  This [notebook](https://www.github.com/notebooks/BluSky%20-%20Getting%20Started.ipynb) demonstrates more features of the 2D transform, including how to visualize the transform to different orders.

<!-- ## Support
---
This effort was supported by [Sandia National Labs](https://www.sandia.gov/).

![Sandia Labs Logo](/images/sandia.png)

with development and maintenance support by [Enthought](https://www.enthought.com).

![Enthought logo](/images/enthought.png) -->
