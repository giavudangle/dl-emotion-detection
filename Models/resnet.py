import keras

backend = keras.backend
layers = keras.layers
models = keras.models
keras_utils = keras.utils

activation = 'elu'


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, kernel_size, padding='same', kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation(activation)(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation(activation)(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, kernel_size, strides=strides, padding='same', kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation(activation)(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    shortcut = layers.Conv2D(filters1, kernel_size, strides=strides, padding='same', kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation(activation)(x)
    return x


def ResNet(include_top=True, weights=None, input_tensor=None, input_shape=None, num_class=None):
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal', name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation(activation)(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, (3, 3), [64, 64], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, (3, 3), [64, 64], stage=2, block='b')
    x = identity_block(x, (3, 3), [64, 64], stage=2, block='c')

    x = conv_block(x, (3, 3), [128, 128], stage=3, block='a')
    x = identity_block(x, (3, 3), [128, 128], stage=3, block='b')
    x = identity_block(x, (3, 3), [128, 128], stage=3, block='c')
    x = identity_block(x, (3, 3), [128, 128], stage=3, block='d')

    x = conv_block(x, (3, 3), [256, 256], stage=4, block='a')
    x = identity_block(x, (3, 3), [256, 256], stage=4, block='b')
    x = identity_block(x, (3, 3), [256, 256], stage=4, block='c')
    x = identity_block(x, (3, 3), [256, 256], stage=4, block='d')
    x = identity_block(x, (3, 3), [256, 256], stage=4, block='e')
    x = identity_block(x, (3, 3), [256, 256], stage=4, block='f')

    x = conv_block(x, (3, 3), [512, 512], stage=5, block='a')
    x = identity_block(x, (3, 3), [512, 512], stage=5, block='b')
    x = identity_block(x, (3, 3), [512, 512], stage=5, block='c')

    if include_top:
        # Classifier 1
        x = layers.Flatten(name='flatten')(x)

        x = layers.Dense(1024, kernel_initializer='he_normal', name='fc1024')(x)
        x = layers.BatchNormalization(name='bn_dense1')(x)
        x = layers.Activation(activation)(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(512, kernel_initializer='he_normal', name='fc512')(x)
        x = layers.BatchNormalization(name='bn_dense2')(x)
        x = layers.Activation(activation)(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(num_class, activation='softmax', name='fc7')(x)

        # Classifier 2
        # x = layers.GlobalAveragePooling2D(name='global_average_pooling')(x)
        #
        # x = layers.Dense(1024, kernel_initializer='he_normal', name='fc1024')(x)
        # x = layers.BatchNormalization(name='bn_dense1')(x)
        # x = layers.Activation(activation)(x)
        #
        # x = layers.Dense(num_class, activation='softmax', name='fc7')(x)

    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name='resnet50')

    if weights is not None:
        model.load_weights(weights)

    return model

# ResNet50(include_top=True, input_shape=(48, 48, 1)).summary()
