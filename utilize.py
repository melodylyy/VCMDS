import numpy as np
import random
from tensorflow.keras.layers import Dense, Input, dot
from tensorflow.keras.models import Model


def BuildModel(train_x,train_y):
    # This returns a tensor
    l = len(train_x[1])
    inputs = Input(shape=(l,))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(64, activation='relu')(inputs)
    x = Dense(16, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_x, train_y,epochs=25)  # starts training
    return model


