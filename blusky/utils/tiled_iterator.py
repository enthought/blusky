from itertools import product

from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing.image.iterator import Iterator
import numpy as np

from traits.api import Array, Bool, Enum, File, HasTraits, Instance, Int, Tuple


def patches(
    test_data,
    use_to_standardize=[],
    tile_size=(64, 64),
    transform_log_scale=-1,
    overlap_log_2=2,
):
    """ Extract a unique (ordered) set of unaugmented patches.
        yields each patch because in some cases the collection
        of all patches could be quite large.

    Parameters
    ----------
    test_data - File
        data to tile in numpy (nh, nw, nchan) layout.
    use_to_standardize - (optional) List(Files)
        one or more data set (above layout) to estimate std/mean.
    transform_log_scale - (optional) Int
        Log-scale of the transfrom (suggested to leave as is)
    overlap_log_2 -  Int
        Log2 overlap of tiles, valid values are 0, 1, 2, ...

    Returns
    -------
    patches - List(Array)
        A complete list of unaugmented tiles from test_data.
    """

    standardize = len(use_to_standardize) > 1

    datagen = TiledDataGenerator(
        featurewise_center=standardize,
        featurewise_std_normalization=standardize,
    )
    if standardize:
        datagen.fit(use_to_standardize)

    tile_iter = TiledIterator(
        twod_image=test_data,
        tile_size=tile_size,
        overlap_log_2=overlap_log_2,
        transform_log_scale=transform_log_scale,
        image_data_generator=datagen,
        shuffle=False,
        batch_size=1,
    )

    for i in range(len(tile_iter._centroids)):
        yield next(tile_iter)[0, ...]


class TiledIterator(HasTraits, Iterator):
    """ Generate tiles from a 2d numpy array. Integrates with Keras image
        generators augmentation is needed.
    """

    #: (M) This is defines the overlap of the tiles, so overlap_log_2 = 0 would
    #  be no overlap,
    #  overlap_log_2 = 1 would be 50% overlap, overlap_log_2 = 2 would be 75%
    #  etc.
    overlap_log_2 = Int(0)

    #: Tile size in samples (height, width)
    tile_size = Tuple((64, 64))

    #: pad the image "reflect", by default set to half the tile size.
    image_padding = Int

    #:
    twod_image = File

    #:
    label_mask = File

    #: 2-d image, tiles from this extracted according the (J,M) defined above
    _twod_image = Array

    #: 2-d label mask, if provided a label is assigned from the center of the
    #  tiles.
    _label_mask = Array

    #: Interpreted as "not labeled", if label mask is provided, tiles
    # associated with this label will be ignored
    null_label = Int(0)

    # Instance of `ImageDataGenerator` to use for random transformations and
    # normalization. If undefined then no augmentation is done, it just creates
    # tiles.
    image_data_generator = Instance(ImageDataGenerator)

    #: number of tiles returned in the batch
    batch_size = Int(32)

    #: Numpy array of sample weights.
    sample_weight = Array

    #: Random seed for rng
    seed = Int(42)

    #: shuffle data before for sampling
    shuffle = Bool(True)

    #: data_format: String, one of `channels_first`, `channels_last`.
    data_format = Enum("channels_first", "channels_last")

    #: floating point format
    dtype = np.float32

    #:
    _centroids = Array

    #: delegate the keras iterator
    _iterator = Instance(Iterator)

    #: options not supported
    save_to_dir = None
    save_prefix = None
    save_format = None
    subset = None

    def __init__(self, **traits):
        """
        """
        HasTraits.__init__(self, **traits)

        self.transform_log_scale = int(
            round(np.log2(np.min(self.tile_size))) - 2
        )

        self._twod_image = np.load(self.twod_image)
        if self.label_mask:
            self._label_mask = np.load(self.label_mask)
            if self._label_mask.shape != self._twod_image.shape:
                raise RuntimeError("label mask/image mismatched dimensions.")

        if self.image_padding < 1:
            self.image_padding = max(self.tile_size) // 2

            self._twod_image = np.pad(
                self._twod_image,
                (self.image_padding, self.image_padding),
                "reflect",
            )
            if self._label_mask.size > 0:
                self._label_mask = np.pad(
                    self._label_mask,
                    (self.image_padding, self.image_padding),
                    "reflect",
                )
        self._set_tile_centers()

        Iterator.__init__(
            self,
            len(self._centroids),
            self.batch_size,
            self.shuffle,
            self.seed,
        )

    def _set_tile_centers(self):
        """
        Generally, loading every tile into memory can be exhaustive, so keep a
        list of centers instead.
        """

        nh, nw = self._twod_image.shape

        # steps in each direction
        dh = max(2 ** (-self.overlap_log_2) * self.tile_size[0], 1)
        dw = max(2 ** (-self.overlap_log_2) * self.tile_size[1], 1)

        i_x = np.arange(self.image_padding, nh - self.image_padding, dh)
        i_y = np.arange(self.image_padding, nw - self.image_padding, dw)

        # centroids
        self._centroids = np.array(list(product(i_x, i_y))).astype(int)

    def _get_tiles(self, index_array):
        """ Load tiles into memory.
        Parameters
        ----------
        index_array - List
            Indices into the "centroids"

        Results 2.1.5
        -------
        tiles - Array
            batch_size (?,tile_size, tile_size)
        """

        nh = self.tile_size[0] // 2
        nw = self.tile_size[1] // 2

        # symmetric about centroid, if tile size is even, adds 1 to the shape
        x = np.array(
            [
                self._twod_image[ix - nh : ix + nh + 1, iy - nw : iy + nw + 1]
                for ix, iy in self._centroids[index_array]
            ]
        )

        # in case we have single channel data
        if len(x.shape) < 4:
            x = np.expand_dims(x, axis=-1)

        return x

    def _get_label(self, index_array):
        """ Optionally get labels.
        Parameters
        ----------
        index_array - List
            Indices into the "centroids"

        Results
        -------
        tiles - Array
            batch_size (?,tile_size, tile_size)
        """

        # symmetric about centroid, if tile size is even, adds 1 to the shape
        x = np.array(
            [
                self._label_mask[ix, iy]
                for ix, iy in self._centroids[index_array]
            ]
        )

        # in case we have single channel data
        if len(x.shape) < 4:
            x = np.expand_dims(x, axis=-1)

        return x

    def _get_batches_of_transformed_samples(self, index_array):
        """ Generate a batch of
        """

        tiles = self._get_tiles(index_array)

        if self.image_data_generator:
            for i in range(len(tiles)):
                x = tiles[i, ...]

                params = self.image_data_generator.get_random_transform(
                    x.shape
                )
                x = self.image_data_generator.apply_transform(
                    x.astype(self.dtype), params
                )
                x = self.image_data_generator.standardize(x)

                tiles[i, ...] = x

        if self.label_mask:
            labels = self._get_label(index_array)
            return (tiles, labels)
        else:
            return tiles


class TiledDataGenerator(ImageDataGenerator):
    """ Extend the Keras ImageDataGenerator to change the standardization method.
    """

    def __init__(self, **kwargs):
        ImageDataGenerator.__init__(self, **kwargs)

    def fit(self, data_to_standardize):
        """ Fit mean/standard deviation.

        Parameters
        ----------
        data_to_standardize - List
            A list of files in npy format to load for standardizing.
        """

        if not isinstance(data_to_standardize, list):
            data_to_standardize = [data_to_standardize]

        # assuming data shape is consistent
        test_data = np.load(data_to_standardize[0])

        if len(test_data.shape) < 3:
            nchan = 1
        elif len(test_data.shape) == 3:
            nchan = test_data.shape[2]
        else:
            raise RuntimeError("required data shape is: (nh, nw, nchan)")

        if self.featurewise_center:
            mn = np.zeros(nchan)
            N = 0

            for i in data_to_standardize:
                val = np.load(i)
                if len(val.shape) < 3:
                    val = np.expand_dims(val, -1)

                N += val.shape[0] * val.shape[1]
                mn += np.array(
                    [np.sum(val[..., j]) for j in range(val.shape[2])]
                )

            self.mean = mn / N

        if self.featurewise_std_normalization:

            # compute the mean again to offset "var":
            mn = np.zeros(nchan)
            N = 0

            for i in data_to_standardize:
                val = np.load(i)
                if len(val.shape) < 3:
                    val = np.expand_dims(val, -1)

                N += val.shape[0] * val.shape[1]
                mn += np.array(
                    [np.sum(val[..., j]) for j in range(val.shape[2])]
                )

            mn /= N

            var = np.zeros(nchan)
            for i in data_to_standardize:
                val = np.load(i)
                if len(val.shape) < 3:
                    val = np.expand_dims(val, -1)

                var += np.array(
                    [np.var(val[..., j] - mn) for j in range(val.shape[2])]
                )

            var /= len(data_to_standardize)
            self.std = np.sqrt(var)

        if self.zca_whitening:
            raise NotImplementedError()
