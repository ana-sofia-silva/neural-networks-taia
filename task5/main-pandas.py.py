import sys
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from matplotlib import pyplot

folder_path = '../animals/'


def define_model():
    model = tf.keras.models.Sequential([
        ResNet50(input_shape=(108,108,3), include_top=False),])
    for layer in model.layers:
        layer.trainable = False
        
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def run_test_harness():
   model = define_model()
   datagen = ImageDataGenerator(rescale=1.0/255.0)
   train_datagen = ImageDataGenerator(fill_mode = 'nearest',validation_split=0.2)  #Split data : 80% train, 20% validation)
   train_generator=train_datagen.flow_from_directory(
        folder_path,
        target_size=(108,108),
        color_mode='rgb',
        class_mode='categorical',
        subset='training')
   validation_generator=train_datagen.flow_from_directory(
        folder_path,
        target_size=(108,108),
        color_mode='rgb',
        class_mode='categorical',
        subset='validation')
   history = model.fit(train_generator,validation_data=validation_generator,epochs=20,verbose=0)



def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

# entry point, run the test harness
run_test_harness()
