
from os import path
import unittest
import numpy as np


class TestAlgorithms(unittest.TestCase):
    def test_morlet2d(self):
        """
        Acid test the implementation using a reference code.
        """
        from blusky.wavelets.morlet2d import Morlet2D
        import blusky.datasets as datasets

        # generated from morlet_2d_noDC.m
        test_wav0 = path.join(
            path.dirname(datasets.__file__), "40Hz_30Hzbw_slant05_0deg.npy"
        )
        test_wav0 = np.load(test_wav0)

        test_wav45 = path.join(
            path.dirname(datasets.__file__), "40Hz_30Hzbw_slant05_45deg.npy"
        )
        test_wav45 = np.load(test_wav45)


        wav = Morlet2D(sample_rate=0.004,
                       center_frequency=45.,
                       bandwidth=(30.,15.),
                       crop=3.5, taper=False)

        _wav0 = wav.kernel(0.0)
        _wav45 = wav.kernel(45.0)

        self.assertTrue(np.max(np.abs(_wav0 - test_wav0)) < 1E-7)
        self.assertTrue(np.max(np.abs(_wav45 - test_wav45)) < 1E-7)
