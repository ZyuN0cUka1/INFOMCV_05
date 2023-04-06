from read_file_stanford_40 import train_set, train_labels, test_set, test_labels, labels
from keras.applications import vgg19
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Flatten, Dense

input_shape = (224, 224, 3)


def residual_block(inputs, filters, stride):
    # Convolutional path
    x = Conv2D(filters, (3, 3), strides=stride, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    # Identity path
    if stride == 2:
        inputs = Conv2D(filters, (1, 1), strides=stride, padding='same')(inputs)
    # Skip connection
    x = Activation('relu')(x + inputs)
    return x


def ResNet18():
    # Input layer
    inputs = Input(shape=input_shape)
    # Convolutional layers
    x = Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    # Residual blocks
    x = residual_block(x, 64, stride=1)
    x = residual_block(x, 64, stride=1)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128, stride=1)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256, stride=1)
    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512, stride=1)
    # Fully connected layers
    x = Flatten()(x)
    x = Dense(labels, activation='softmax')(x)
    # Create model
    _model = Model(inputs=inputs, outputs=x)
    return _model


def model_create():
    # rn = vgg19.VGG19(include_top=True,
    #                  weights='imagenet',
    #                  input_tensor=None,
    #                  input_shape=None,
    #                  pooling=None,
    #                  classes=1000,
    #                  classifier_activation='softmax')
    rn = ResNet18()
    rn.summary()
    rn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return rn


def plot_training_results(history, model_name):
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title(model_name + ' Training and Validation Loss and Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    model = model_create()
    model.summary()
    history0 = model.fit(train_set, train_labels, validation_data=(test_set, test_labels), epochs=15)
    model.save('Model/Stanford40.h5')
