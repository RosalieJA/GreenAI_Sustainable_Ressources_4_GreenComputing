#INSTALLATION DES BIBLIOTHEQUES
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.cluster import MiniBatchKMeans
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn import metrics
import sys
import sklearn
from codecarbon import EmissionsTracker

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# CONVERSION DE CHAQUE IMAGE EN UN VECTEUR DE DIMENSION 1
X = x_train.reshape(len(x_train),-1)
Y = y_train

# NORMALISATION DES DONNEES (0 - 1)
X = X.astype(float) / 255.

print(X.shape)
print(X[0].shape)

n_digits = len(np.unique(y_test))
print(n_digits)

# INITIALISATION DU MODELE KMeans
kmeans = MiniBatchKMeans(n_clusters = n_digits)

# AJUSTER LE MODELE AUX DONNEES D'APPRENTISSAGE
kmeans.fit(X)
kmeans.labels_


def infer_cluster_labels(kmeans, actual_labels):
    """
    Associe l'étiquette la plus probable à chaque grappe dans le modèle KMeans
    retourne : dictionnaire des grappes assignées à chaque étiquette
    """

    inferred_labels = {}

    for i in range(kmeans.n_clusters):

        # Trouver l'index des points dans le cluster
        labels = []
        index = np.where(kmeans.labels_ == i)

        # Ajouter les étiquettes pour chaque point du cluster
        labels.append(actual_labels[index])

        # Déterminer l'étiquette la plus courante
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assigner le cluster à une valeur dans le dictionnaire inferred_labels
        if np.argmax(counts) in inferred_labels:
            # ajouter le nouveau numéro au tableau existant à ce slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # Créer un nouveau tableau dans ce slot
            inferred_labels[np.argmax(counts)] = [i]

        #print(labels)
        #print('Cluster: {}, label: {}'.format(i, np.argmax(counts)))

    return inferred_labels

def infer_data_labels(X_labels, cluster_labels):
    """
    Détermine l'étiquette de chaque tableau, en fonction du groupe auquel il a été assigné.
    retourne : étiquettes prédites pour chaque tableau
    """

    # tableau vide de taille len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)

    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key

    return predicted_labels


# Test des fonctions infer_cluster_labels() et infer_data_labels()
cluster_labels = infer_cluster_labels(kmeans, Y)
X_clusters = kmeans.predict(X)
predicted_labels = infer_data_labels(X_clusters, cluster_labels)
print (predicted_labels[:20])
print (Y[:20])


# Calcul et affichage des métriques
def calculate_metrics(estimator, data, labels):
    print('Number of Clusters: {}'.format(estimator.n_clusters))
    print('Inertia: {}'.format(estimator.inertia_))
    print('Homogeneity: {}'.format(metrics.homogeneity_score(labels, estimator.labels_)))

clusters = [10, 16, 36, 64, 144, 256]

# Test de différents nombres de clusters
for n_clusters in clusters:
    tracker = EmissionsTracker()
    tracker.start()
    estimator = MiniBatchKMeans(n_clusters = n_clusters)
    estimator.fit(X)

    # Affichage des metrics du cluster
    calculate_metrics(estimator, X, Y)

    # Détermination des étiquettes prédites
    cluster_labels = infer_cluster_labels(estimator, Y)
    predicted_Y = infer_data_labels(estimator.labels_, cluster_labels)

    # Calcul et affichage de la précision (accuracy)
    print('Accuracy: {}\n'.format(metrics.accuracy_score(Y, predicted_Y)))
    emissions = tracker.stop()
    print(f"Emissions : {emissions} kg CO2")

# Test de l'algorithme des kmeans sur l'ensemble de données de test
X_test = x_test.reshape(len(x_test),-1)
X_test = X_test.astype(float) / 255.

tracker = EmissionsTracker()
tracker.start()
# Initialisation et adaptation de l'algorithme KMeans aux données d'apprentissage
kmeans = MiniBatchKMeans(n_clusters = 256)
kmeans.fit(X)
cluster_labels = infer_cluster_labels(kmeans, Y)
emissions = tracker.stop()
print(f"Emissions : {emissions} kg CO2")
    
# Prédiction des étiquettes pour les données de test
test_clusters = kmeans.predict(X_test)
print(type(X_test))
predicted_labels = infer_data_labels(kmeans.predict(X_test), cluster_labels)

# Calcul et affichage de la précision
print('Accuracy: {}\n'.format(metrics.accuracy_score(y_test, predicted_labels)))

#Test sur nos images
files = ["zero.png", "one.png", "two.png", "three.png", "four.png", "five.png", "six.png", "seven.png"]

for file in files:
    img = load_img("in/" + file, color_mode='grayscale', target_size=(28, 28))
    x = img_to_array(img).reshape(1, -1).astype(float) / 255.0
    cluster = kmeans.predict(x)
    predicted_label = infer_data_labels(kmeans.predict(x), cluster_labels)
    print(file + " => " + str(predicted_label) + "\n")
