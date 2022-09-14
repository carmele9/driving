from keras.preprocessing import image
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, Dense, Flatten, MaxPooling2D,BatchNormalization
import tensorflow
import random,shutil
import os



def generator(dir, gen = image.ImageDataGenerator(rescale=1./255), shuffle= True, batchsize = 1, target_size = (24,24), classmode = "categorical"):
    return gen.flow_from_directory(dir, shuffle= shuffle, batch_size = batchsize, target_size = target_size, class_mode = classmode, color_mode="grayscale")


BS = 32
TS = (24,24)
data_train = generator("drowssines/data/train", shuffle = True, batchsize= BS, target_size= TS)
data_test = generator("drowssines/data/test", shuffle = True, batchsize= BS, target_size= TS)
SPE =  len(data_train.classes)//BS
VS = len(data_test)//BS
model = Sequential([
    Conv2D(32, kernel_size= (3,3), activation= "relu", input_shape= (24,24,1)),
    MaxPooling2D(pool_size= (1,1)),
    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(pool_size= (1,1)),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(pool_size= (1,1)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation = "relu"),
    Dropout(0.5),
    Dense(2,activation="softmax")

])

model.compile(optimizer= "adam", loss="categorical_crossentropy", metrics= ["accuracy"])
model.fit(data_train,validation_data=data_test,epochs=15, steps_per_epoch=SPE, validation_steps=VS)
#model.save("drowssines/models/trial")

