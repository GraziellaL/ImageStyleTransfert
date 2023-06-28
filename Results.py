"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Fichier d'utilisation de la lib pour les résultats fournis dans le rapport
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Import des librairies utilitaires
import matplotlib.pyplot as plt
import glob
from keras.backend import mean
# Imports utiles de la lib ImageSytleTransfert
from ImageStyleTransfert import ImageStyleTransfert, IMG_SIZE

# Couches utilisées dans l'article
article_content_layer = "block2_conv2"
article_style_layers = ["block1_conv1", "block2_conv1", "block3_conv1",
                        "block4_conv1", "block5_conv1"]


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Comparaison des comportements de ContentReconstruction et StyleReconstruction
    en fonction du VGG utilisé : type de pooling, normalisation des poids du réseau
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def pooling_normalisation_comparison(operation="style"):
    memo = []
    weight_norm_list = [False, True, False, True]
    pooling_type_list = ["max", "max", "avg", "avg"]
    for norm, pool in zip(weight_norm_list, pooling_type_list):
        if operation == "content":
            if norm:
                file_name = "ContentReconstruction_VGG19_" + pool + "_norm"
            else:
                file_name = "ContentReconstruction_VGG19_" + pool
            reconstruction = ImageStyleTransfert(operation="ContentReconstruction",
                                                 weights_normalization=norm,
                                                 pooling_type=pool,
                                                 content_layer=article_content_layer)
            reconstruction.apply(input_content_image="Islande.jpg",
                                 generated_image_file_name=file_name)

            print(" Vérification de la normalisation du reseau ")
            print("content_feature moyenne", float(mean(reconstruction.content_feature, axis=(0, 1, 2))))
        else:
            if norm:
                file_name = "StyleReconstruction_VGG19_" + pool + "_norm"
            else:
                file_name = "StyleReconstruction_VGG19_" + pool
            reconstruction = ImageStyleTransfert(operation="StyleReconstruction",
                                                 weights_normalization=norm,
                                                 pooling_type=pool,
                                                 style_layers=article_style_layers)
            reconstruction.apply(input_style_image="Style1.jpg",
                                 generated_image_file_name=file_name)

            print(" Vérification de la normalisation du reseau ")
            for feature in reconstruction.style_features:
                print("style_features moyenne", float(mean(feature[0], axis=(0, 1, 2))))
        memo.append(reconstruction.display_values)

    for i, values in enumerate(memo):
        if i == 0:
            legend = "max"
        elif i == 1:
            legend = "max_norm"
        elif i == 2:
            legend = "avg"
        elif i == 3:
            legend = "avg_norm"
        if values["style_loss"]["all_values"]:
            plt.plot(values["style_loss"]["all_values"][1:], label=legend)
        if values["content_loss"]["all_values"]:
            plt.plot(values["content_loss"]["all_values"][1:], label=legend)
    plt.legend(loc='upper right')
    if values["style_loss"]["all_values"]:
        plt.title("Convergence de l'erreur de style")
    if values["content_loss"]["all_values"]:
        plt.title("Convergence de l'erreur de contenu")
    plt.show()

pooling_normalisation_comparison(operation= "content")
# pooling_normalisation_comparison(operation= "style")


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Comparaison de "L-BFGS" et "Adam"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def optimizers_comparison(operation="style"):

    optimizers_list = ["L-BFGS", "Adam"]
    memo = []
    for opt in optimizers_list:
        if operation == "content":
            reconstruction = ImageStyleTransfert(operation="ContentReconstruction",
                                                 content_layer=article_content_layer)
            reconstruction.apply(input_content_image="Islande.jpg",
                                 optimizer_type=opt)

            memo.append(reconstruction.display_values)
        else:
            reconstruction = ImageStyleTransfert(operation="StyleReconstruction",
                                                 style_layers=article_style_layers)
            reconstruction.apply(input_style_image="Style1.jpg",
                                 optimizer_type=opt)

            memo.append(reconstruction.display_values)

    for i, values in enumerate(memo):
        if values["style_loss"]["all_values"]:
            plt.plot(values["style_loss"]["all_values"][1:], label=f"{optimizers_list[i]}")
        if values["content_loss"]["all_values"]:
            plt.plot(values["content_loss"]["all_values"][1:], label=f"{optimizers_list[i]}")
    plt.legend(loc='upper right')
    if values["style_loss"]["all_values"]:
        plt.title("Convergence de l'erreur de style")
    if values["content_loss"]["all_values"]:
        plt.title("Convergence de l'erreur de contenu")
    plt.show()

# optimizers_comparison(operation="style")


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    ContentReconstruction et StyleReconstruction vue d'ensemble - Figure 1 de l'article
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def layers_comparison(operation="style"):

    content_layer_list = ["block1_conv2", "block2_conv2", "block3_conv2", "block4_conv2", "block5_conv2"]
    style_layers_list = [["block1_conv1"], ["block1_conv1", "block2_conv1"],
                         ["block1_conv1", "block2_conv1", "block3_conv1"],
                         ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1"],
                         ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]]
    memo = []
    if operation == "content":
        for content_layer in content_layer_list:
            file_name = f"ContentReconstruction_{content_layer}"
            reconstruction = ImageStyleTransfert(operation="ContentReconstruction",
                                                 content_layer=content_layer)
            reconstruction.apply(input_content_image="Islande.jpg",
                                 generated_image_file_name=file_name)

            memo.append(reconstruction.display_values)
    else:
        for style_layers in style_layers_list:
            file_name = f"StyleReconstruction_{style_layers}"
            reconstruction = ImageStyleTransfert(operation="StyleReconstruction",
                                                 style_layers=style_layers)
            reconstruction.apply(input_style_image="Style1.jpg",
                                 generated_image_file_name=file_name)

            memo.append(reconstruction.display_values)

    for i, values in enumerate(memo):
        if values["style_loss"]["all_values"]:
            plt.plot(values["style_loss"]["all_values"][1:], label=f"{style_layers_list[i]}")
        if values["content_loss"]["all_values"]:
            plt.plot(values["content_loss"]["all_values"][1:], label=f"{content_layer_list[i]}")
    plt.legend(loc='upper right')
    if values["style_loss"]["all_values"]:
        plt.title("Convergence de l'erreur de style")
    if values["content_loss"]["all_values"]:
        plt.title("Convergence de l'erreur de contenu")
    plt.show()

# layers_comparison(operation= "content")
# layers_comparison(operation= "style")


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Rapport de alpha / beta - Figure 4 de l'article
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def alpha_beta_contribution():
    style_weight_list = [1, 1e1, 1e2, 1e3, 1e4]
    memo = []
    for weight in style_weight_list:
        file_name = f"StyleTransfert_Islande_Style1_Style_weight_impact_{weight}"
        style_transfert = ImageStyleTransfert(operation="StyleTransfert",
                                              content_layer=article_content_layer,
                                              style_layers=article_style_layers,
                                              content_weight=1, style_weight=weight)
        style_transfert.apply(input_content_image="Islande.jpg", input_style_image="Style1.jpg",
                              generated_image_file_name=file_name)
        memo.append(style_transfert.display_values)

# alpha_beta_contribution()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Effet des différentes couches - Figure 5 de l'article
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def layers_impact_comparison():
    content_layer = [article_content_layer, "block4_conv2"]
    for content_l in content_layer:
        file_name = f"StyleTransfert_Islande_Style1_layers_impact_{content_l}"
        style_transfert = ImageStyleTransfert(operation="StyleTransfert",
                                              content_layer=content_l,
                                              style_layers=article_style_layers)
        style_transfert.apply(input_content_image="Islande.jpg", input_style_image="Style1.jpg",
                              generated_image_file_name=file_name)

# layers_impact_comparison()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Image d’initialisation - Figure 6 de l'article
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def initialisation_impact_comparison():
    init_list = ["content", "style"]
    for init in init_list:
        file_name = f"StyleTransfert_Islande_Style1_init_impact_{init}"
        style_transfert = ImageStyleTransfert(operation="StyleTransfert",
                                              content_layer=article_content_layer,
                                              style_layers=article_style_layers)
        style_transfert.apply(input_content_image="Islande.jpg", input_style_image="Style1.jpg",
                              image_init_type=init,
                              generated_image_file_name=file_name)

# initialisation_impact_comparison()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Style transfert vue d'ensemble
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def style_transfert_overview():
    styles_files = list(glob.iglob('Inputs/StyleImages/*.jpg'))
    content_files = list(glob.iglob('Inputs/ContentImages/*.jpg'))
    for styles_file in styles_files:
        styles_file = styles_file[len("Inputs/StyleImages/"):]
        for content_file in content_files:
            content_file = content_file[len("Inputs/ContentImages/"):]
            print(f"Style transfert appliqué sur {styles_file} et {content_file}")
            style_weight_list = [50, 250, 500]
            for weight in style_weight_list:
                file_name = f"StyleTransfert_{content_file}_{styles_file}_weight_{weight}"
                style_transfert = ImageStyleTransfert(operation="StyleTransfert",
                                                      content_layer=article_content_layer,
                                                      style_layers=article_style_layers,
                                                      content_weight=1, style_weight=weight)
                style_transfert.apply(input_content_image=content_file, input_style_image=styles_file,
                                      generated_image_file_name=file_name)

# style_transfert_overview()
