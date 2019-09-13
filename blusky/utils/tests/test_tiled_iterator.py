import numpy as np
from os import path
import unittest

import blusky.datasets as datasets
from blusky.utils.tiled_iterator import TiledIterator, TiledDataGenerator


class TestTiledIterator(unittest.TestCase):
    def setUp(self):

        self.test_file_1 = path.join(
            path.dirname(datasets.__file__), "twod_image_1.npy"
        )
        self.test_file_2 = path.join(
            path.dirname(datasets.__file__), "twod_image_2.npy"
        )

        self.test_data_1 = np.load(
            path.join(path.dirname(datasets.__file__), "twod_image_1.npy")
        )
        self.test_data_2 = np.load(
            path.join(path.dirname(datasets.__file__), "twod_image_2.npy")
        )

    def test_tiled_iterator_nogen(self):
        """ test tiled iterator without the generator"""
        tile_no_gen = TiledIterator(
            twod_image=self.test_file_1, overlap_log_2=0
        )
        tile = next(tile_no_gen)

        shape = tile.shape

        # defaults
        self.assertTrue(shape[0] == 32)
        self.assertTrue(shape[1] == 65)
        self.assertTrue(shape[2] == 65)
        self.assertTrue(shape[3] == 1)

        #
        img0 = self.test_data_1[0:65, 0:65]
        np.array_equal(tile, img0)

        # no overlap
        tile = next(tile_no_gen)
        img0 = self.test_data_1[65 : 2 * 65, 65 : 2 * 65]
        np.array_equal(tile, img0)

        # --- overlapping --- #
        tile_no_gen = TiledIterator(
            twod_image=self.test_file_1, overlap_log_2=2
        )

        tile = next(tile_no_gen)

        shape = tile.shape

        # defaults
        self.assertTrue(shape[0] == 32)
        self.assertTrue(shape[1] == 65)
        self.assertTrue(shape[2] == 65)
        self.assertTrue(shape[3] == 1)

        #
        img0 = self.test_data_1[0:65, 0:65]
        np.array_equal(tile, img0)

        # 64/(2**2) = 16
        tile = next(tile_no_gen)
        img0 = self.test_data_1[16 : 16 + 65, 16 : 16 + 65]
        np.array_equal(tile, img0)

    def test_tiled_data_generator(self):
        """ Add image generator, standardize"""

        datagen = TiledDataGenerator(
            featurewise_center=True, featurewise_std_normalization=True
        )

        datagen.fit(self.test_file_2)

        mn = np.mean(self.test_data_2)
        std = np.std(self.test_data_2)

        self.assertAlmostEqual(mn / datagen.mean[0], 1, places=6)
        self.assertAlmostEqual(std / datagen.std[0], 1, places=6)

        tile_gen = TiledIterator(
            twod_image=self.test_file_1,
            overlap_log_2=1,
            image_data_generator=datagen,
        )

        next(tile_gen)
        next(tile_gen)
        tile = next(tile_gen)

        # 64//2**1 = 32
        ofst = 32 * 2
        img0 = self.test_data_1[ofst : ofst + 65, ofst : ofst + 65]

        np.allclose(tile, (img0 - mn) / std)
