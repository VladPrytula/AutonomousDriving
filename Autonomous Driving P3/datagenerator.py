from imageutils import *
from keras.preprocessing.image import *
from dataiterators import RegressionDirectoryIterator


class ImageGenerator(object):
    """Generates minibatches with a real-time data augmentation.

    Based on the version of the ImageDataGenerator from Keras
    (https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py).
    """

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=0.,
                 rotation_value_transform=None,
                 width_shift_range=0.,
                 width_shift_value_transform=None,
                 height_shift_range=0.,
                 height_shift_value_transform=None,
                 shear_range=0.,
                 shear_value_transform=None,
                 zoom_range=0.,
                 zoom_value_transform=None,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 horizontal_flip_value_transform=None,
                 vertical_flip=False,
                 vertical_flip_value_transform=None,
                 rescale=None,
                 dim_ordering='default',
                 cropping=(0, 0, 0, 0)):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.__dict__.update(locals())
        self.mean = None
        self.std = None
        self.principal_components = None
        self.rescale = rescale

        if dim_ordering not in {'tf', 'th'}:
            raise Exception('dim_ordering should be "tf" (channel after row and '
                            'column) or "th" (channel before row and column). '
                            'Received arg: ', dim_ordering)
        self.dim_ordering = dim_ordering
        if dim_ordering == 'th':
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3
        if dim_ordering == 'tf':
            self.channel_index = 3
            self.row_index = 1
            self.col_index = 2

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise Exception('zoom_range should be a float or '
                            'a tuple or list of two floats. '
                            'Received arg: ', zoom_range)

        self.cropping = cropping

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return RegressionNumpyArrayIterator(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

    def flow_from_directory(self, directory, values,
                            target_size=(256, 256), color_mode='rgb',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix='', save_format='jpeg'):
        return RegressionDirectoryIterator(
            directory, values, self,
            target_size=target_size, color_mode=color_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

    def crop(self, x):
        return crop_image(x, self.cropping)

    def standardize(self, x):
        if self.rescale:
            if callable(self.rescale):
                x = self.rescale(x)
            else:
                x *= self.rescale

        # x is a single image, so it doesn't have image number at index 0
        img_channel_index = self.channel_index - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_index, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_index, keepdims=True) + 1e-7)

        if self.featurewise_center:
            x -= self.mean
        if self.featurewise_std_normalization:
            x /= (self.std + 1e-7)

        if self.zca_whitening:
            flatx = np.reshape(x, (x.size))
            whitex = np.dot(flatx, self.principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))

        return x

    def random_transform(self, x, y):
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1

        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.rotation_value_transform:
            y = self.rotation_value_transform(y, theta)

        if self.height_shift_range:
            px = np.random.uniform(-self.height_shift_range, self.height_shift_range)
            tx = px * x.shape[img_row_index]
        else:
            tx = 0

        if self.height_shift_value_transform:
            y = self.height_shift_value_transform(y, px)

        if self.width_shift_range:
            py = np.random.uniform(-self.width_shift_range, self.width_shift_range)
            ty = py * x.shape[img_col_index]
        else:
            ty = 0

        if self.width_shift_value_transform:
            y = self.width_shift_value_transform(y, py)

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.shear_value_transform:
            y = self.shear_value_transform(y, shear)

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        if self.zoom_value_transform:
            y = self.zoom_value_transform(y, zx, zy)

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        h, w = x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)
        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_index)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
                if self.horizontal_flip_value_transform:
                    y = self.horizontal_flip_value_transform(y)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)
                if self.vertical_flip_value_transform:
                    y = self.vertical_flip_value_transform(y)

        return x, y

    def fit(self, X,
            augment=False,
            rounds=1,
            seed=None):
        if seed is not None:
            np.random.seed(seed)

        X = np.copy(X)
        cropped = np.zeros((X.shape[0], get_cropped_shape(X.shape[1:], self.cropping)))
        for i, img in enumerate(X):
            cropped[i] = self.crop(img)
        X = cropped

        if augment:
            aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
            for r in range(rounds):
                for i in range(X.shape[0]):
                    aX[i + r * X.shape[0]] = self.random_transform(X[i])
            X = aX

        if self.featurewise_center:
            self.mean = np.mean(X, axis=0)
            X -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=0)
            X /= (self.std + 1e-7)

        if self.zca_whitening:
            flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
            sigma = np.dot(flatX.T, flatX) / flatX.shape[0]
            U, S, V = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)
