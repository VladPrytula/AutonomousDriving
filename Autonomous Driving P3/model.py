import json
import numpy as np

from keras.callbacks import ModelCheckpoint

import config
from clonebuilder import *


np.random.seed(1)


if __name__ == '__main__':
    model = get_model()
    model.summary()
    model.compile(optimizer='adam', loss='mse')

    # Persist trained model
    model_json = model.to_json()
    with open('model.json', 'w') as f:
        json.dump(model_json, f)

    rdi_train, rdi_val = get_generator(TRAIN_DATA, VALIDATION_DATA, True, batch_size=BATCH_SIZE)

    checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto')

    # Train the model with exactly one version of each image
    history = model.fit_generator(rdi_train,
                                  samples_per_epoch=rdi_train.N,
                                  validation_data=rdi_val,
                                  nb_val_samples=rdi_val.N,
                                  nb_epoch=NB_EPOCH,
                                  callbacks=[checkpoint])
