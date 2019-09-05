import unittest

from keras.layers import Input

from blusky.transforms.cascade_2d import Cascade2D
from blusky.transforms.cascade_tree import CascadeTree
from blusky.transforms.default_decimation import DefaultDecimation

class TestAlgorithms(unittest.TestCase):
    def test_default_decimation(self):        
        """ test the default decimation method.
        """
        import numpy as np

        cascade_tree = CascadeTree(Input(shape=(99,99,1)),
                                   order=3)
        cascade_tree.generate([1,2,3], lambda x,y,z : x)
        
        # go down a path in the tree
        root_node = cascade_tree.root_node
        first_layer = root_node.children[0]
        second_layer = first_layer.children[0]
        third_layer = second_layer.children[0]        

        deci = DefaultDecimation(oversampling=0)
        wav = np.empty((99,99))
        wavp = deci.decimate_wavelet(wav, 2)
        np.array_equal(wav[::2,::2], wavp)
        
        
        # if I would decimate wavelet/conv
        deci = DefaultDecimation(oversampling=0)

        w, c = deci.resolve_scales(root_node)
        self.assertTrue(w == 1 and c == 1)
        w, c = deci.resolve_scales(first_layer)        
        self.assertTrue(w == 1 and c == 2)
        w, c = deci.resolve_scales(second_layer)        
        self.assertTrue(w == 2 and c == 4)
        w, c = deci.resolve_scales(third_layer)        
        self.assertTrue(w == 4 and c == 8)

        #
        deci = DefaultDecimation(oversampling=1)

        w, c = deci.resolve_scales(root_node)
        self.assertTrue(w == 1 and c == 1)
        w, c = deci.resolve_scales(first_layer)        
        self.assertTrue(w == 1 and c == 1)
        w, c = deci.resolve_scales(second_layer)        
        self.assertTrue(w == 1 and c == 2)
        w, c = deci.resolve_scales(third_layer)        
        self.assertTrue(w == 2 and c == 4)
        
        #
        deci = DefaultDecimation(oversampling=2)

        w, c = deci.resolve_scales(root_node)
        self.assertTrue(w == 1 and c == 1)
        w, c = deci.resolve_scales(first_layer)        
        self.assertTrue(w == 1 and c == 1)
        w, c = deci.resolve_scales(second_layer)        
        self.assertTrue(w == 1 and c == 1)
        w, c = deci.resolve_scales(third_layer)        
        self.assertTrue(w == 1 and c == 2)

        
