import numpy as np

from traits.api import (
    Any,
    Enum,
    Function,
    HasStrictTraits,
    Instance,
    Int,
    List,
    Str,
    Tuple,
)

from keras.layers import Layer


class Node(HasStrictTraits):
    """
   The nodes of the cascade tree keep track of the recipe
   of the convolution and the scale.
   """

    #: Name of the layer
    name = Str
    #: Keras layer name of the node
    layer_name = Str
    #: Scale of the transform at this node
    scale = Int
    #: Node whence I came.
    parent = Any
    #: Nodes are my children
    children = List
    #: function that does something with the data in the node
    payload = Any

    def __str__(self):
        return "( %s, %d )" % (self.name, self.scale)


class CascadeTree(HasStrictTraits):
    """
   Generatates a tree of wavelet convolutions to some order
   in the scattering transform.
   """

    #: the order of the transform
    order = Int
    #: The lowest scale origin of hte transform,
    #  expect this is typically the input image.
    root_node = Instance(Node)

    def __init__(self, input_layer, **traits):
        self.root_node = Node(name="x", layer_name="x", payload=input_layer)
        super().__init__(**traits)

    def generate(self, wavelet_bank, conv_function):
        """
       Generates a tree of wavelet convolutions to some
       order:
       Parameters
       ----------
       wavelet_bank - List(Array)
           A list of wavelets as a 2-d array.
       input_name - Str
           The name of the tree.
       """

        current_layer = [self.root_node]
        for stage in np.arange(self.order + 1):
            next_layer = []
            for current_node in current_layer:
                for i, wv in enumerate(wavelet_bank):
                    scale = i + 1
                    if scale > current_node.scale:
                        new_name = "|%s*psi_%d|" % (current_node.name, scale)
                        layer_name = "%s-psi_%d" % (current_node.layer_name, scale)
                        new_node = Node(
                            name=new_name,
                            layer_name=layer_name,
                            scale=scale,
                            parent=current_node,
                        )
                        lhs = current_node.payload
                        new_node.payload = conv_function(lhs, wv, layer_name)
                        current_node.children.append(new_node)
                        next_layer.append(new_node)
            current_layer = next_layer

    # layer by layer output
    def display(self):
        current_layer = [self.root_node]
        for stage in np.arange(self.order + 1):
            next_layer = []
            print("Layer %d" % stage)
            for current_node in current_layer:
                print(current_node)
                next_layer += current_node.children
            current_layer = next_layer

    # need a method to return a list of all (non root) nodes
    def get_convolutions(self):
        current_layer = [self.root_node]
        all_convolutions = []
        # starting at the first layer, don't need the root node (input)
        for stage in np.arange(1, self.order + 1):
            next_layer = []
            for current_node in current_layer:
                next_layer += current_node.children

            current_layer = next_layer
            all_convolutions += [n.payload for n in current_layer]
        return all_convolutions
