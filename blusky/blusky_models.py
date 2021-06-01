from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Input, UpSampling1D

from blusky.transforms.apply_father_wavelet_1d import ApplyFatherWavlet1D
from blusky.transforms.cascade_1d import Cascade1D
from blusky.transforms.cascade_tree import CascadeTree
from blusky.transforms.default_decimation import DefaultDecimation
from blusky.utils.pad_1d import Pad1D
from blusky.wavelets.wavelet_factories_1d import (vanilla_gabor_1d, 
                                                  vanilla_morlet_1d,
                                                  calibrate_wavelets_1d)

def apply_upsampling(layer, N):
    n = layer.shape[1]
    factor = int(N//n)
    if(factor > 0):
        return UpSampling1D(factor)(layer)
    else:
        return layer

def build_model_1d(N, J, order,
                   oversampling=2, 
                   conv_padding="valid", 
                   concatenate=False,
                   apply_father_wavelet=True):
    """ Build the 1-d wavelet transform
    shape - tuple, length of the 1-d data (maybe mulitvariate)
    J - int, the scale of the transform.
    order - int, the order of the transform.
    oversampling - int, 2**oversampling 
            delays decimation of the series.
    conv_padding - "valid" or "same" supplied to (see Keras' Conv1D)
            "valid" should be faster.
    concatenate - True/False, whether or not to concatenate the output.
    apply_father_wavelet - True/False, the final step in the transform is to convolve each output
    """
    sample_rate = 1.0
    
    inp = Input(shape=(N,1))
    
    wavelets = [vanilla_morlet_1d(sample_rate, J, j=i) 
                        for i in range(0,J)]
    calibrate_wavelets_1d(wavelets)

    father_wavelet = vanilla_gabor_1d(sample_rate, J)

    deci = DefaultDecimation(oversampling=oversampling)

    # pad
    pad_1d = Pad1D(wavelets, decimation=deci, 
                   conv_padding=conv_padding)
    padded = pad_1d.pad(inp)

    cascade_tree = CascadeTree(padded, order=order)

    cascade = Cascade1D(decimation=deci, _padding=conv_padding)
    convs = cascade.transform(cascade_tree, wavelets=wavelets)

    # Create layers to remove padding
    cascade_tree = CascadeTree(padded, order=order)
    cascade_tree.generate(wavelets, pad_1d.unpad)
    unpad = cascade_tree.get_convolutions()

    # Remove the padding
    unpadded_convs = [i[1](i[0]) for i in zip(convs, unpad)]

    if apply_father_wavelet:
        appl = ApplyFatherWavlet1D(wavelet=father_wavelet, 
                                   J=J, 
                                   img_size=(N,), 
                                   sample_rate=sample_rate)    

        sca_transf = appl.convolve(unpadded_convs)
    
        if concatenate:
            sca_transf = Concatenate()(sca_transf)
    
        # implement scattering transform.
        model = Model(inputs=inp, outputs=sca_transf)
    else:
        if concatenate:
            unpadded_convs = [apply_upsampling(i,N) for i in unpadded_convs]
            unpadded_convs = Concatenate()(unpadded_convs)
        
        model = Model(inputs=inp, outputs=unpadded_convs)
    
    return model