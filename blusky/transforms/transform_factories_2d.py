from keras.layers import Input
from keras.models import Model
import numpy as np

from blusky.transforms.apply_father_wavelet_2d import ApplyFatherWavlet2D
from blusky.transforms.cascade_2d import Cascade2D
from blusky.transforms.cascade_tree import CascadeTree
from blusky.transforms.default_decimation import DefaultDecimation
from blusky.utils.pad_2d import Pad2D
from blusky.utils.visualize_2d import PlotElement, Visualize2D

from blusky.wavelets.wavelet_factories_2d import (
    vanilla_gabor_2d,
    vanilla_morlet_2d,
)


def vanilla_scattering_transform(
    J,
    img_size,
    sample_rate,
    overlap_log_2=0,
    order=2,
    oversampling=1,
    num_angles=8,
        do_father=True
):

    # to reproduce scatnet.m and kymatio definitions (see NOTICE.txt)
    angles = tuple(
        [
            90.0
            - np.rad2deg(
                (int(num_angles - num_angles / 2 - 1) - theta)
                * np.pi
                / num_angles
            )
            for theta in range(num_angles)
        ]
    )

    # vanilla filter bank
    wavelets = [vanilla_morlet_2d(sample_rate, j=i) for i in range(0, J)]
    father_wavelet = vanilla_gabor_2d(sample_rate, j=J)

    # method of decimation
    deci = DefaultDecimation(oversampling=oversampling)

    # input
    inp = Input(shape=img_size)

    # valid padding
    cascade2d = Cascade2D("none", 0, decimation=deci, angles=angles)

    # Pad the input
    pad_2d = Pad2D(wavelets, decimation=deci)
    padded = pad_2d.pad(inp)

    # Apply cascade with successive decimation
    cascade_tree = CascadeTree(padded, order=order)
    cascade_tree.generate(wavelets, cascade2d._convolve)
    convs = cascade_tree.get_convolutions()

    # Create layers to remove padding
    cascade_tree = CascadeTree(padded, order=order)
    cascade_tree.generate(wavelets, pad_2d._unpad_same)
    unpad = cascade_tree.get_convolutions()

    # Remove the padding
    unpadded_convs = [i[1](i[0]) for i in zip(convs, unpad)]

    # Complete the scattering transform with the father wavelet
    if do_father:
        apply_conv = ApplyFatherWavlet2D(
            J=J-1,
            overlap_log_2=overlap_log_2,
            img_size=img_size,
            sample_rate=sample_rate,
            wavelet=father_wavelet,
        )
        sca_transf = apply_conv.convolve(unpadded_convs)
        model = Model(inputs=inp, outputs=sca_transf)

    else:
        model = Model(inputs=inp, outputs=unpadded_convs)
        

    # generate visuals too, it's a factory
    cascade_tree = CascadeTree(padded, order=order)
    cascade_tree.generate(wavelets, cascade2d._convolve)

    root_element = PlotElement(name=cascade_tree.root_node.name)
    root_element.radius_range = (0.02, 1)
    root_element.angle_range = (0, 180)

    viz = Visualize2D(angles=angles)
    viz.recurse(cascade_tree.root_node, root_element, max_order=order)

    return model, viz
