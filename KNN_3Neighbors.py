#INSTALLATION DES BIBLIOTHEQUES
import ssl
import certifi
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import tensorflow as tf
import time  # Importation du module time
from codecarbon import EmissionsTracker

ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())

# RECOLTE DES DONNEES
# chargement des données MNIST (ensemble d'entraînement et de test)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalisation des données
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0

# Diviser les données en ensembles de formation et de test
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Standardisation des données
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# Initialisation du tracker CodeCarbon
tracker = EmissionsTracker(gpu_ids=[])

# CREATION DU MODELE KNN
knn = KNeighborsClassifier(n_neighbors=3)  #on peut ajuster le nombre de voisins

# ENTRAINEMENT DU MODELE
print("Début de l'entraînement du modèle KNN")
tracker.start()  # Démarrage du suivi des émissions
knn.fit(x_train, y_train)
emissions_train = tracker.stop()  # Arrêt du suivi des émissions
print(f"Émissions pendant l'entraînement: {emissions_train} kg CO2")

# EVALUATION DU MODELE
print("Évaluation du modèle sur les données de validation")
tracker.start()  # Démarrage du suivi des émissions pour l'évaluation
y_val_pred = knn.predict(x_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
emissions_val = tracker.stop()  # Arrêt du suivi des émissions
print(f"\nValidation accuracy: {val_accuracy}")
print(f"Émissions pendant l'évaluation sur validation: {emissions_val} kg CO2")

print("Évaluation du modèle sur les données de test")
tracker.start()  # Démarrage du suivi des émissions pour l'évaluation
y_test_pred = knn.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
emissions_test = tracker.stop()  # Arrêt du suivi des émissions
print(f"\nTest accuracy: {test_accuracy}")
print(f"Émissions pendant l'évaluation sur test: {emissions_test} kg CO2")

# PREDICTION
files = ["zero.png", "one.png", "two.png", "three.png", "four.png", "five.png", "six.png", "seven.png"]
for file in files:
    img = load_img("in/" + file, color_mode='grayscale', target_size=(28, 28))
    x = img_to_array(img)
    x = x.reshape(1, 28*28)
    x = scaler.transform(x / 255.0)

    # Mesure du temps de prédiction
    start_time = time.time()
    y = knn.predict(x)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Prédiction de la classe
    bestClass = y[0]
    print(f"1/1 ━━━━━━━━━━━━━━━━━━━━ 0s {int(elapsed_time * 1000)}ms/step")
    print(f"{file} => {bestClass}\n")
