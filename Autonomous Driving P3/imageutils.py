import scipy.misc as spm
from keras.preprocessing.image import *


def normalize(images, new_max, new_min, old_max=None, old_min=None):
    if old_min is None:
        old_min = np.min(images)
    if old_max is None:
        old_max = np.max(images)

    return (images - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min


def crop_image(img, cropping):
    return img[cropping[0]:img.shape[0] - cropping[1], cropping[2]:img.shape[1] - cropping[3], :]


def get_cropped_shape(img_shape, cropping):
    return (img_shape[0] - cropping[0] - cropping[1],
            img_shape[1] - cropping[2] - cropping[3],
            img_shape[2])


def resize_image(img, size):
    return spm.imresize(img, size)


def extract_filename(path):
    return path.split('/')[-1]


def adjust_path(path, new_location):
    return '%s/%s' % (new_location, extract_filename(path))


def load_images(paths, target_size):
    images = np.zeros((len(paths), target_size, 3))
    for i, p in enumerate(paths):
        img = load_img(p, target_size=target_size)
        img = img_to_array(img, dim_ordering='tf')
        images[i] = img

    return images

