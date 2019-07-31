from os import path
import unittest
import numpy as np


class TestAlgorithms(unittest.TestCase):
    def test_vanilla_factory(self):
        """
        Test data generated from the Kymatio libray:
        https://github.com/kymatio/kymatio
        """
        import blusky.datasets as datasets
        from blusky.wavelets.wavelet_factories_2d import vanilla_gabor_2d

        ang = np.pi / 3.55

        wavj0 = vanilla_gabor_2d(0.001, j=0, xi=2.35619, crop=30.7 * 4).kernel(
            np.rad2deg(ang)
        )
        wavj1 = vanilla_gabor_2d(0.001, j=1, xi=2.35619, crop=30.7 * 2).kernel(
            np.rad2deg(ang)
        )
        wavj2 = vanilla_gabor_2d(0.001, j=2, xi=2.35619, crop=30.7).kernel(
            np.rad2deg(ang)
        )

        test_wav0 = path.join(
            path.dirname(datasets.__file__), "vanilla_gabor_0j.npy"
        )
        test_wav0 = np.load(test_wav0)

        test_wav1 = path.join(
            path.dirname(datasets.__file__), "vanilla_gabor_1j.npy"
        )
        test_wav1 = np.load(test_wav1)

        test_wav2 = path.join(
            path.dirname(datasets.__file__), "vanilla_gabor_2j.npy"
        )
        test_wav2 = np.load(test_wav2)

        relative_error0 = (
            0.5
            * np.max(np.abs(wavj0 - test_wav0))
            / np.max(np.abs(wavj0 + test_wav0))
        )
        self.assertTrue(relative_error0 < 1e-5)

        relative_error1 = (
            0.5
            * np.max(np.abs(wavj1 - test_wav1))
            / np.max(np.abs(wavj1 + test_wav1))
        )
        self.assertTrue(relative_error1 < 1e-5)

        relative_error2 = (
            0.5
            * np.max(np.abs(wavj2 - test_wav2))
            / np.max(np.abs(wavj2 + test_wav2))
        )
        self.assertTrue(relative_error2 < 1e-5)
