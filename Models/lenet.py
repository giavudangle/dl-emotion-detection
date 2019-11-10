from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras import backend as K
from keras.regularizers import l2
from keras.layers import BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D


# activation: tanh, elu, relu
# kernel_initializer: glorot_uniform, he_normal

class LeNet:
    @staticmethod
    def build(numChannels, imgRows, imgCols, numClasses, activation="relu", kernel_initializer="he_normal",
              weightsPath=None):
        num_filter = 64

        inputShape = (numChannels, imgRows, imgCols)

        if K.image_data_format() == 'channels_last':
            inputShape = (imgRows, imgCols, numChannels)

        model = Sequential()

        # Conv1
        model.add(Conv2D(num_filter, kernel_size=(5, 5), padding="same", kernel_initializer=kernel_initializer,
                         input_shape=inputShape))
        model.add(BatchNormalization())
        model.add(Activation(activation=activation))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        # Conv2
        model.add(Conv2D(num_filter * 2, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer))
        model.add(BatchNormalization())
        model.add(Activation(activation=activation))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        # Conv3
        model.add(Conv2D(num_filter * 2 * 2, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer))
        model.add(BatchNormalization())
        model.add(Activation(activation=activation))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        # Conv4
        model.add(Conv2D(num_filter * 2 * 2 * 2, kernel_size=(3, 3), padding="same",
                         kernel_initializer=kernel_initializer))
        model.add(BatchNormalization())
        model.add(Activation(activation=activation))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        # Classifier

        model.add(Flatten())

        model.add(Dense(1024, kernel_initializer=kernel_initializer))
        model.add(BatchNormalization())
        model.add(Activation(activation=activation))
        model.add(Dropout(0.5))

        model.add(Dense(512, kernel_initializer=kernel_initializer))
        model.add(BatchNormalization())
        model.add(Activation(activation=activation))
        model.add(Dropout(0.5))

        # Classifier 1

        # model.add(GlobalAveragePooling2D())
        #
        # model.add(Dense(1024, kernel_initializer=kernel_initializer))
        # model.add(BatchNormalization())
        # model.add(Activation(activation=activation))

        # Classifier 2

        # model.add(GlobalMaxPooling2D())
        #
        # model.add(Dense(1024, kernel_initializer=kernel_initializer))
        # model.add(BatchNormalization())
        # model.add(Activation(activation=activation))

        model.add(Dense(numClasses, activation="softmax", kernel_initializer=kernel_initializer))

        if weightsPath is not None:
            print("[INFO] load model...")
            model.load_weights(weightsPath)

        return model

# LeNet.build(1, 48, 48, 7).summary()
