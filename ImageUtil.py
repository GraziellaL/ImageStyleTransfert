"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Image Style Transfer Using Convolutional Neural Networks
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Import des librairies utilitaires
import os
import matplotlib.pyplot as plt
import numpy as np
# Imports pour le chargement et visualisation des images
from keras.utils import load_img, img_to_array, array_to_img

# Taille des images choisie pour les calculs
IMG_SIZE = 224  # 512 (article) - 224 (imagenet)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Chargement et visualisation des images 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def load_image(directory_path, image_filename, dimension):
    """
    Fonction de chargement de l'image et conversion en array
    :param directory_path:
    :param image_filename:
    :param dimension:
    """
    image = load_img(os.path.join(directory_path, image_filename),
                     grayscale=False, target_size=(dimension, dimension))
    image = img_to_array(image, data_format="channels_last")
    return image


def plot_image(image, title=None):
    """
    Fonction d'affichage des images
    :param image:
    :param title:
    """
    image = array_to_img(image, data_format="channels_last")
    plt.imshow(image)
    if title is not None:
        plt.title(title)


def resize_flatten_image(input_image):
    """
    Fonction de remise en forme de l'image applatie
    Utile pour l'algo d'optimisation L-BFGS
    :param input_image:
    """
    if len(input_image.shape) == 1:  # pour le cas flatten
        input_image = input_image.reshape((IMG_SIZE, IMG_SIZE, 3))
        input_image = input_image.astype(np.float32)  # float32 pour L-BFGS pour les modeles Keras
    return input_image
