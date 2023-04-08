import read_file_stanford_40 as rst
from midframe import train_data, train_labels, test_data, test_labels
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Flatten, Dense, GlobalAveragePooling2D, Dropout
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import glob
import cv2
import numpy as np
import tensorflow as tf

def plot_accuracy(history, model_name):
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title(model_name + ' Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_loss(history, model_name):
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(model_name + ' Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


model_path = 'Model/Stanford40.h5'
pretrained_model = tf.keras.models.load_model(model_path)


num_classes = 12
new_output = tf.keras.layers.Dense(num_classes, activation='softmax')(pretrained_model.layers[-2].output)
pretrained_model = tf.keras.Model(inputs=pretrained_model.input, outputs=new_output)


learning_rate = 0.001  
optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=learning_rate, momentum=0.9)

pretrained_model.compile(optimizer=optimizer,
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])


print(pretrained_model.input_shape)

train_data = train_data / 255.0
test_data = test_data / 255.0

print(train_data.shape)
print(test_data.shape)

epochs = 30  
history = pretrained_model.fit(train_data, train_labels,
                               validation_data=(test_data, test_labels),
                               epochs=epochs)
pretrained_model.save('Model/hmdb51frame.h5')
test_loss, test_acc = pretrained_model.evaluate(test_data, test_labels)
print(f'Test accuracy: {test_acc}')
plot_accuracy(history, 'hmdb51frame')
plot_loss(history, 'hmdb51frame')