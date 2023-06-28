"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Image Style Transfer Using Convolutional Neural Networks
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Import des librairies utilitaires
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
# Imports pour le chargement et visualisation des images
from keras.utils import save_img
# Imports pour les différentes techniques d'optimisations
from keras.optimizers import Adam
from scipy.optimize import fmin_l_bfgs_b

# Imports utiles de la lib ImageSytleTransfert
from ImageUtil import plot_image, load_image, resize_flatten_image, IMG_SIZE
from VGGModelManager import VGGModelManager


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Définitions des fonction de perte
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def squared_error_loss(f, p):
    """
    Equation n°1
    :param f:
    :param p:
    """
    return 1 / 2 * tf.reduce_sum(tf.square(f - p))  # Equation (1)


def gram_matrix(f, area, depth):
    """
    Equation n°3
    :param f:
    :param area:
    :param depth:
    """
    f = tf.reshape(f, (area, depth))
    g = tf.matmul(tf.transpose(f), f)
    return g


def layer_style_contribution(current_style_feature, style_feature):
    """
    Equation n°4
    :param current_style_feature:
    :param style_feature:
    """
    # Cartes cartéristiques de dim M
    m_area = current_style_feature.shape[0] * current_style_feature.shape[1]
    # Profondeur de la couche = Nombre de cartes
    n_depth = current_style_feature.shape[2]

    g_current_style = gram_matrix(current_style_feature, m_area, n_depth)  # Gram matrix de dim NxN
    a_style = gram_matrix(style_feature, m_area, n_depth)

    coeff = 1 / (2.0 * (n_depth ** 2) * (m_area ** 2))
    e_contribution = coeff * squared_error_loss(g_current_style, a_style)

    # print("Style - Contribution de la couche : ", current_style_feature.shape,
    #       f"{float(e_contribution):.1E}")
    return e_contribution


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   Classe de mise en oeuvre de la méthode de style transfert présentée dans l'article
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class ImageStyleTransfert:

    def __init__(self, operation="StyleTransfert",
                 weights_normalization=True, pooling_type="avg",
                 content_layer=None, style_layers=None,
                 content_weight=1, style_weight=1e2):
        # Opération à réaliser :
        self.operation = operation

        # Vérification de la cohérence des données d'entrées
        assert self.operation in ["StyleTransfert", "ContentReconstruction", "StyleReconstruction"]
        if operation == "StyleTransfert":
            assert content_layer is not None and style_layers is not None
        elif self.operation == "StyleReconstruction":
            assert style_layers is not None
        elif self.operation == "ContentReconstruction":
            assert content_layer is not None

        # Définition du modèle de récupération des cartes caratéristiques
        self.weights_normalization = weights_normalization
        self.pooling_type = pooling_type
        self.model = VGGModelManager(content_layer=content_layer, style_layers=style_layers,
                                     pooling=self.pooling_type,
                                     weights_normalization=self.weights_normalization)

        # Poids des fonctions de perte liée au contenu et au style
        self.style_weight = style_weight  # beta dans l'article
        self.content_weight = content_weight  # alpha dans l'article

        self.generated_image = None
        self.content_image_name = None  # utiles pour la sauvegarde des images générés
        self.style_image_name = None
        self.content_image = None
        self.style_image = None
        self.content_feature =  None
        self.style_features = None

        # Mémorisation utile pour l'algorithme L-BFGS
        self.current_grad = None

        # Mémorisation pour affichage de la convergence de l'algorithme d'optimisation
        self.display_values = {"content_loss": {"first_value": [], "all_values": []},
                               "style_loss": {"first_value": [], "all_values": []},
                               "total_loss": {"first_value": [], "all_values": []}}

    def init_content_and_style_images(self, input_content_image, input_style_image):
        """
        Chargement, visualisation et pré-processing des données d'entrées
        :param input_content_image:
        :param input_style_image:
        """
        # Image de contenu :
        content_image = load_image("Inputs/ContentImages/", input_content_image,
                                   IMG_SIZE) if input_content_image is not None else None
        # Image du style :
        style_image = load_image("Inputs/StyleImages/", input_style_image,
                                 IMG_SIZE) if input_style_image is not None else None

        # Visualisation des images
        if input_content_image:
            plt.subplot(1, 2, 1)
            plot_image(content_image, "Image de contenu")
        if input_style_image:
            plt.subplot(1, 2, 2)
            plot_image(style_image, "Image de style")
        plt.suptitle("Visualisation des images")
        plt.show()

        # Pré-processing
        self.content_image = self.model.pre_process(content_image) if input_content_image is not None else None
        self.style_image = self.model.pre_process(style_image) if input_style_image is not None else None

    def loss_memorisation(self, current_loss, loss_type):
        """
        loss_memorisation des loss
        :param current_loss:
        :param loss_type:
        """
        if len(self.display_values[loss_type]["first_value"]) < 5:
            self.display_values[loss_type]["first_value"].append(float(current_loss))
            # print(f"{loss_type} first_value : {float(current_loss):.1E}")
        else:
            first_value = min(self.display_values[loss_type]["first_value"])
            if first_value == 0:
                first_value = max(self.display_values[loss_type]["first_value"])

            self.display_values[loss_type]["all_values"].append(float(current_loss)*100/first_value)
            # print(f"{loss_type} : ", round(float(current_loss)*100/first_value, 2))

    def content_loss(self, current_image):
        """
        Fonction de perte du contenu
        Equation n°1
        :param current_image:
        """
        current_content_feature = self.model.get_content_feature(current_image)
        content_loss = squared_error_loss(current_content_feature, self.content_feature)  # Equation (1)

        self.loss_memorisation(content_loss, "content_loss")

        return content_loss

    def style_loss(self, current_image):
        """
        Fonction de perte du style
        Equation n°5
        :param current_image:
        """
        current_style_features = self.model.get_style_features(current_image)

        weight_layer = 1 / len(self.model.style_layers)

        # Augmentation de l'échelle d'erreur sans quoi devant les trop petites valeurs de loss
        # l'algorithme L-BFGS converge beaucoup trop vite
        if self.weights_normalization:
            weight_layer *= 1e5

        style_loss = 0
        for current_style_feature, style_feature in zip(current_style_features,
                                                        self.style_features):
            if len(current_style_feature.shape) > 3:
                current_style_feature = current_style_feature[0]
                style_feature = style_feature[0]
            layer_contribution = layer_style_contribution(current_style_feature, style_feature)
            style_loss += weight_layer * layer_contribution  # Equation (5)

        self.loss_memorisation(style_loss, "style_loss")

        return style_loss

    def total_loss(self, current_image):
        """
        Fonction de perte à minimiser pour le transfert de style
        :param current_image:
        """
        content_loss = self.content_loss(current_image)
        style_loss = self.style_loss(current_image)
        total_loss = self.content_weight * content_loss + self.style_weight * style_loss  # Equation (7)

        self.loss_memorisation(total_loss, "total_loss")

        # print(f"style_loss : {float(style_loss):.1E}, content_loss : {float(content_loss):.1E}")
        # print(f"total_loss : {float(total_loss):.1E}")
        content_vs_style_contrib = float((self.content_weight * content_loss) / (self.style_weight * style_loss))
        # print("content_weight*content_loss / self.style_weight*style_loss : ",
        #       f"{content_vs_style_contrib:.1E}")

        return total_loss

    def compute_grads_and_loss(self, current_image):
        """
        Calcul simultané de l'erreur et du gradient
        :param current_image:
        """
        loss_function = None
        if self.operation == "StyleTransfert":
            loss_function = self.total_loss
        elif self.operation == "ContentReconstruction":
            loss_function = self.content_loss
        elif self.operation == "StyleReconstruction":
            loss_function = self.style_loss

        with tf.GradientTape() as tape:
            total_loss = loss_function(current_image)
        grad = tape.gradient(total_loss, current_image)

        return grad, total_loss

    def compute_loss(self, current_image):
        """
        Fonction d'erreur pour l'optimisation L-BFGS
        :param current_image:
        """
        current_image = resize_flatten_image(current_image)
        current_image = np.expand_dims(current_image, axis=0)
        current_image = tf.Variable(current_image)
        grad, total_loss = self.compute_grads_and_loss(current_image)
        total_loss = np.float64(total_loss)  # float64 pour L-BFGS
        # Mémorisation du gradient
        self.current_grad = np.float64(grad.numpy().squeeze(0).flatten())
        return total_loss

    def compute_grad(self, current_image):
        """
        Gradient pour l'optimisation L-BFGS
        Renvoie le gradient mémorisé, permet d'éviter de faire 2 fois le calcul
        :param current_image:
        """
        return self.current_grad

    def init_generated_image(self, init_type):
        """
        Initialisation de l'image
        :param init_type:
        """
        if init_type == "random":
            # Initialisation aléatoire
            img = np.random.uniform(0, 255, (IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)
            rand_image = self.model.pre_process(img)
            return rand_image

        elif init_type == "content":
            # Initialisation avec l"image de contenu
            return self.content_image

        elif init_type == "style":
            # Initialisation avec l"image de style
            return self.style_image

    def adam_optimization(self, max_iter):
        """
        optimisation avec adam
        :param max_iter:
        """
        # learning_rate testés 1, 10, 100, 200, 300, 1000
        # Pas de convergence pour 1 et 10, Instabilités pour 300, Divergence pour 1000
        # epsilon testés 1, 1e-1
        # beta_1 testés 0.99, 0.9
        optimizer = Adam(learning_rate=200, beta_1=0.99, epsilon=1e-1)
        for i in range(max_iter):
            grads, loss = self.compute_grads_and_loss(self.generated_image)
            optimizer.apply_gradients([(grads, self.generated_image)])
            clipped = tf.clip_by_value(self.generated_image, -self.model.norm_means,
                                       255 - self.model.norm_means)
            self.generated_image.assign(clipped)

    def l_bfgs_optimization(self, max_iter):
        """
        optimisation avec l_bfgs
        :param max_iter:
        """
        self.generated_image = np.float64(self.generated_image.numpy().flatten())

        # Bornes de l'image :
        max_bound = 255 - min(self.model.norm_means)
        min_bound = -max(self.model.norm_means)
        bounds = np.ones((len(self.generated_image), 2), dtype=np.float64)*[min_bound, max_bound]

        self.generated_image, loss, info = fmin_l_bfgs_b(self.compute_loss,
                                                         self.generated_image,
                                                         fprime=self.compute_grad,
                                                         maxiter=max_iter,
                                                         bounds=bounds)
        return loss

    def convergence_display(self):
        """
        Affichage des courbes d'évolution des différents loss au cours de l'optimisation
        """

        if self.display_values["content_loss"]["all_values"]:
            plt.plot(self.display_values["content_loss"]["all_values"],
                     color="red", label="content_loss")
        if self.display_values["style_loss"]["all_values"]:
            plt.plot(self.display_values["style_loss"]["all_values"],
                     color="blue", label="style_loss")
        if self.display_values["total_loss"]["all_values"]:
            plt.plot(self.display_values["total_loss"]["all_values"],
                     color="green", label="total_loss")
        plt.legend(loc="upper right")
        plt.title("Convergences de l'erreur totale")
        plt.show()

    def apply(self, input_content_image=None, input_style_image=None, image_init_type="random",
              optimizer_type="L-BFGS", num_iterations=100, generated_image_file_name=None):
        """
        Application de la méthode de StyleTransfert (ou StyleReconstruction ou ContentReconstruction)
        :param input_content_image:
        :param input_style_image:
        :param image_init_type:
        :param optimizer_type:
        :param num_iterations:
        :param generated_image_file_name:
        """

        # Vérification de la cohérence des données d'entrée
        if self.operation == "StyleTransfert":
            assert input_content_image is not None and input_style_image is not None
        elif self.operation == "StyleReconstruction":
            assert input_style_image is not None
        elif self.operation == "ContentReconstruction":
            assert input_content_image is not None

        start_time = time.time()

        # Initialisation des images de contenu et de style
        self.content_image_name = input_content_image
        self.style_image_name = input_style_image
        self.content_image = None
        self.style_image = None
        self.init_content_and_style_images(input_content_image, input_style_image)
        # Récupération des représentations caractéristiques de contenu et de style
        self.content_feature = self.model.get_content_feature(
            self.content_image) if input_content_image is not None else None
        self.style_features = self.model.get_style_features(
            self.style_image) if input_style_image is not None else None

        # Initalisation de l'image et visualisation
        assert image_init_type in ["random", "content", "style"]
        self.generated_image = self.init_generated_image(image_init_type)
        plot_generated_image = self.model.post_process(self.generated_image)
        plot_image(plot_generated_image, "Image initiale")
        plt.show()

        # Optimisation
        assert optimizer_type in ["L-BFGS", "Adam"]
        if optimizer_type == "Adam":
            self.adam_optimization(num_iterations)
        elif optimizer_type == "L-BFGS":
            self.l_bfgs_optimization(num_iterations)

        # Temps de calcul
        print(f"Durée de la génération : {time.time()-start_time}")

        # Visualisation de la convergence de l'algorithme:
        self.convergence_display()

        # Sauvegarde de l'image générée et visualisation
        if generated_image_file_name:
            fname = generated_image_file_name + ".jpg"
        else:
            content_name = self.content_image_name[:len(self.content_image_name)-4] if self.content_image_name else ""
            style_name = self.style_image_name[:len(self.style_image_name)-4] if self.style_image_name else ""
            fname = f"{self.operation}_{content_name}_{style_name}_{optimizer_type}.jpg"
        plot_generated_image = self.model.post_process(self.generated_image)
        save_img("Results/" + fname, plot_generated_image)
        plot_image(plot_generated_image, f"{self.operation} - Meilleure image générée")
        plt.show()
