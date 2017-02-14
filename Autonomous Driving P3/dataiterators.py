from imageutils import *
from keras.preprocessing.image import *


class CustomNumpyArrayIterator(Iterator):
    """
        This implementation is a modified version of the NumpyArrayIterator from Keras
        (https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py).
    """

    def __init__(self, X, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        if y is not None and len(X) != len(y):
            raise Exception('X (images tensor) and y (labels) '
                            'should have the same length. '
                            'Found: X.shape = %s, y.shape = %s' % (np.asarray(X).shape, np.asarray(y).shape))
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.X = X
        self.y = y
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(CustomNumpyArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        output_shape = get_cropped_shape(self.X[0].shape, self.image_data_generator.cropping)
        batch_x = np.zeros((current_batch_size,) + output_shape)
        batch_y = np.zeros(current_batch_size)
        for i, j in enumerate(index_array):
            x = self.X[j]
            y = self.y[j]
            x = self.image_data_generator.crop(x)
            x, y = self.image_data_generator.random_transform(x, y)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_y[i] = y
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        return batch_x, batch_y


class RegressionDirectoryIterator(Iterator):
    """
        This implementation is a modified version of the DirectoryIterator from Keras
        (https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py).
    """

    def __init__(self, paths, values, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering='default',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.paths = paths
        self.values = values
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)

        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')

        self.color_mode = color_mode
        self.dim_ordering = dim_ordering

        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        self.nb_sample = len(paths)
        self.nb_values = len(values)
        if self.nb_sample != self.nb_values:
            raise ValueError("Number of values=%d does not match "
                             "number of samples=%d" % (self.nb_values, self.nb_sample))

        super(RegressionDirectoryIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        output_shape = get_cropped_shape(self.image_shape, self.image_data_generator.cropping)
        batch_x = np.zeros((current_batch_size,) + output_shape)
        batch_y = np.zeros(current_batch_size)
        grayscale = self.color_mode == 'grayscale'

        # build batch of image data
        for i, j in enumerate(index_array):
            path = self.paths[j]
            img = load_img(path, grayscale=grayscale, target_size=self.target_size)

            y = self.values[j]
            x = img_to_array(img, dim_ordering=self.dim_ordering)
            x = self.image_data_generator.crop(x)
            x, y = self.image_data_generator.random_transform(x, y)
            x = self.image_data_generator.standardize(x)

            batch_x[i] = x
            batch_y[i] = y

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        return batch_x, batch_y
