"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Classe de gestion du modèle
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Import des librairies utilitaires
import numpy as np
import tensorflow as tf

# Imports la mise en place du modèle
from keras.applications import VGG19
from keras.models import Model

# Imports utiles de la lib ImageSytleTransfert
from VGG19_avg import VGG19_avg
from ImageUtil import resize_flatten_image, IMG_SIZE


class VGGModelManager:

    def __init__(self, content_layer=None, style_layers=None,
                 pooling="avg", weights_normalization=True):
        # Paramètres du reseau VGG utilisé pour récupérer les cartes caractéristiques :
        self.pooling = pooling
        self.weights_normalization = weights_normalization

        # Définition du réseau VGG
        self.vgg_network = self.init_vgg()

        # Choix des couches utilisées pour récupérer le contenu et/ou le style
        self.content_layers = [content_layer] if content_layer else []
        self.style_layers = style_layers if style_layers else []

        # Définition du modèle
        self.model = self.init_model()

        # Valeurs moyennes des données ImageNet
        self.norm_means = np.array([103.939, 116.779, 123.68])

    def init_vgg(self):
        """
        Initialisation du réseau VGG en fonction des paramètres de choix du pooling et
        de normalisation de poids du réseau
        """
        if self.pooling == "max":
            if not self.weights_normalization:
                vgg = VGG19(include_top=False, weights="imagenet",
                            input_shape=(IMG_SIZE, IMG_SIZE, 3))
            else:
                vgg = VGG19(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
                vgg.load_weights('vgg19_norm_weights.h5')
            vgg.trainable = False
        elif self.pooling == "avg":
            vgg = VGG19_avg(weights_normalization=self.weights_normalization)
            vgg.trainable = False

        return vgg

    def init_model(self):
        """
        Initialisation du modèle
        """
        # Récupération des cartes caractéristiques pour le calcul des loss de content et de style
        content_outputs = [self.vgg_network.get_layer(name).output for name in self.content_layers]
        style_outputs = [self.vgg_network.get_layer(name).output for name in self.style_layers]
        model_outputs = content_outputs + style_outputs

        # Création du modèle
        model = Model(inputs=self.vgg_network.input, outputs=model_outputs)
        model.trainable = False
        return model

    def pre_process(self, input_image):
        """
        Fonction de pré-processing des images
        :param input_image:
        """
        # Même processus de pré-traitement que celui utilisé pour l’entraînement des réseaux VGG :
        # Le pré-processing des images consiste à soustraire la moyenne des données ImageNet
        # pre_processed_image = vgg19.preprocess_input(pre_processed_image, data_format="channels_last")
        # Traitement équivalent plus clair à comprendre pour le post-process
        input_image[:, :, 0] -= self.norm_means[0]
        input_image[:, :, 1] -= self.norm_means[1]
        input_image[:, :, 2] -= self.norm_means[2]

        # Conversion RGB / BGR
        pre_processed_image = input_image[:, :, ::-1]

        # Mise en forme pour l'utilisation des algo de tf
        pre_processed_image = np.expand_dims(pre_processed_image, axis=0)
        pre_processed_image = tf.Variable(pre_processed_image)

        return pre_processed_image

    def post_process(self, input_image):
        """
        Fonction de post-processing des images
        :param input_image:
        """
        # Remises en formes
        if isinstance(input_image, tf.Variable):
            input_image = input_image.numpy()
        if input_image.shape == (1, IMG_SIZE, IMG_SIZE, 3):
            input_image = np.squeeze(input_image, 0)
        input_image = resize_flatten_image(input_image)

        # Conversion BGR / RGB
        input_image = input_image[:, :, ::-1]

        # Traitement inverse de la moyenne des données ImageNet
        input_image[:, :, 0] += self.norm_means[0]
        input_image[:, :, 1] += self.norm_means[1]
        input_image[:, :, 2] += self.norm_means[2]

        # Clip values to 0-255
        post_processed_image = np.clip(input_image, 0, 255)

        return post_processed_image

    def get_content_feature(self, input_image):
        """
        Récupération de la couche de contenu
        :param input_image:
        """
        content_feature = self.model(input_image)[0]
        return content_feature

    def get_style_features(self, input_image):
        """
        Récupération des couches de style
        :param input_image:
        """
        style_features = self.model(input_image)[len(self.content_layers):]
        return style_features
