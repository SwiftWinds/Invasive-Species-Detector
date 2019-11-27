from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import models
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np


# starting point
my_model = models.Sequential()

# Add first convolutional block
my_model.add(Conv2D(16, (3, 3), activation='relu', padding='same',
                    input_shape=(224, 224, 3)))
my_model.add(MaxPooling2D((2, 2), padding='same'))

# second block
my_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
my_model.add(MaxPooling2D((2, 2), padding='same'))
# third block
my_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
my_model.add(MaxPooling2D((2, 2), padding='same'))
# fourth block
my_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
my_model.add(MaxPooling2D((2, 2), padding='same'))

# global average pooling
my_model.add(GlobalAveragePooling2D())
# fully connected layer
my_model.add(Dense(64, activation='relu'))
my_model.add(BatchNormalization())
# make predictions
my_model.add(Dense(2, activation='sigmoid'))

# Show a summary of the model. Check the number of trainable parameters
my_model.summary()

# use early stopping to optimally terminate training through callbacks
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# save best model automatically
mc = ModelCheckpoint("C:/Users/aeshon/Desktop/11-16-19model.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)

cb_list = [mc, es]


# compile model
my_model.compile(optimizer='adam', loss='binary_crossentropy',
                 metrics=['accuracy'])

# set up data generator
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

# get batches of training images from the directory
train_generator = data_generator.flow_from_directory(
        'C:/Users/aeshon/Desktop/train500',
        target_size=(224, 224),
        batch_size=10,
        class_mode='categorical')

# get batches of validation images from the directory
validation_generator = data_generator.flow_from_directory(
        'C:/Users/aeshon/Desktop/test200',
        target_size=(224, 224),
        batch_size=10,
        class_mode='categorical')


history = my_model.fit_generator(
        train_generator,
        epochs=5,
        steps_per_epoch=2000,
        validation_data=validation_generator,
        validation_steps=1000, callbacks=cb_list)

my_model.save("C:/Users/aeshon/Desktop/11-16-19model.h5")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylim([.5,1.1])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("C:/Users/aeshon/Desktop/11-16-19model.png", dpi=300)
