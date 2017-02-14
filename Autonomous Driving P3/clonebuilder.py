from datagenerator import ImageGenerator
from imageutils import load_images, get_cropped_shape
from config import SHIFT_OFFSET, SHIFT_RANGE, CROPP_FACTOR, IMG_SIZE
from keras.applications.vgg16 import VGG16
from keras.layers import Convolution2D, Input, Dropout
from keras.layers import Flatten, Dense
from keras.models import Model
import pandas as pd
from config import *


def rescale_input():
    return lambda x: x / 127.5 - 1.


def flip_tranform():
    return lambda val: -val


def shift_transform():
    return lambda val, shift: val - (
        (SHIFT_OFFSET / SHIFT_RANGE) * shift)


def get_model():
    image_shape = [IMG_SIZE[0], IMG_SIZE[1], 3]
    input_layer = Input(shape=get_cropped_shape(image_shape, CROPP_FACTOR))
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_layer)

    # Remove the last block of the VGG16 net.
    [base_model.layers.pop() for _ in range(4)]
    base_model.outputs = [base_model.layers[-1].output]
    base_model.layers[-1].outbound_nodes = []

    # Make sure pre trained layers from the VGG net don't change while training.
    for layer in base_model.layers:
        layer.trainable = False

    # Add last block to the VGG model with modified sub sampling.
    layer = base_model.outputs[0]
    layer = Convolution2D(512, 3, 3, subsample=(2, 2), activation='relu', border_mode='same', name='block5_conv1')(
        layer)
    layer = Convolution2D(512, 3, 3, subsample=(1, 2), activation='relu', border_mode='same', name='block5_conv2')(
        layer)
    layer = Convolution2D(512, 3, 3, subsample=(1, 2), activation='relu', border_mode='same', name='block5_conv3')(
        layer)

    layer = Flatten()(layer)
    layer = Dropout(.2)(layer)
    layer = Dense(2048, activation='relu', name='fc1')(layer)
    layer = Dropout(.2)(layer)
    layer = Dense(1024, activation='relu', name='fc2')(layer)
    layer = Dropout(.5)(layer)
    layer = Dense(1, activation='linear', name='predictions')(layer)

    return Model(input=base_model.input, output=layer)


def get_generator(train, validation, from_directory=False, batch_size=32, fit_sample_size=None):
    header = ['center_img', 'left_img', 'right_img', 'steering_angle', 'throttle', 'break', 'speed']

    train_info = pd.concat([pd.read_csv(path, names=header) for path in train])
    validation_info = pd.concat([pd.read_csv(path, names=header) for path in validation])

    # Create feature value pairs for the left camera images by subtracting the offset from the steering angle.
    left = train_info[['left_img', 'steering_angle']].copy()
    left.loc[:, 'steering_angle'] -= SHIFT_OFFSET

    # Create feature value pairs for the right camera images by adding the offset to the steering angle.
    right = train_info[['right_img', 'steering_angle']].copy()
    right.loc[:, 'steering_angle'] += SHIFT_OFFSET

    paths = pd.concat([train_info.center_img, left.left_img, right.right_img]).str.strip()
    values = pd.concat([train_info.steering_angle, left.steering_angle, right.steering_angle])

    train_data_generator = ImageGenerator(rescale=rescale_input(),
                                          horizontal_flip=True,
                                          channel_shift_range=0.2,
                                          width_shift_range=SHIFT_RANGE,
                                          width_shift_value_transform=shift_transform(),
                                          horizontal_flip_value_transform=flip_tranform(),
                                          cropping=CROPP_FACTOR)

    validation_data_generator = ImageGenerator(rescale=rescale_input(),
                                               cropping=CROPP_FACTOR)

    if fit_sample_size is not None:
        sample_to_fit = load_images(paths.sample(fit_sample_size), IMG_SIZE)
        train_data_generator.fit(sample_to_fit)
        validation_data_generator.fit(sample_to_fit)
        del sample_to_fit

    if from_directory:
        return (
            train_data_generator.flow_from_directory(paths.values, values.values, shuffle=True,
                                                     target_size=IMG_SIZE,
                                                     batch_size=batch_size),
            validation_data_generator.flow_from_directory(validation_info.center_img.values,
                                                          validation_info.steering_angle.values,
                                                          shuffle=True,
                                                          target_size=IMG_SIZE, batch_size=batch_size))
    else:
        images = load_images(paths, IMG_SIZE)
        val_images = load_images(validation_info.center_img, IMG_SIZE)

        return (train_data_generator.flow(images, values.values, shuffle=True, batch_size=batch_size),
                validation_data_generator.flow(val_images, validation_info.steering_angle.values, shuffle=True,
                                               batch_size=batch_size))
