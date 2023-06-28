"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    VGG19 avec AveragePooling2D
    Fortement inspiré de keras.applications.VGG19
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Imports pour la définition de la fonction VGG avec pooling moyen
from keras import backend
from keras.applications import imagenet_utils
from keras.engine import training
from keras.layers import VersionAwareLayers
from keras.utils import data_utils

# Imports utiles de la lib ImageSytleTransfert
from ImageUtil import IMG_SIZE

layers = VersionAwareLayers()

WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                       'keras-applications/vgg19/'
                       'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')


def VGG19_avg(weights_normalization=True):
    """
    VGG19 avec AveragePooling2D
    :param weights_normalization:
    """
    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        None,
        default_size=IMG_SIZE,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=False,
        weights=None)

    img_input = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = layers.GlobalAveragePooling2D()(x)

    inputs = img_input
    # Create model.
    model = training.Model(inputs, x, name='vgg19_avg')

    # Load weights.
    if weights_normalization:
        weights_path = "vgg19_avg_norm_weights.h5"
    else:
        weights_path = data_utils.get_file(
            'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='253f8cb515780f3b799900260a226db6')
    model.load_weights(weights_path)

    return model
