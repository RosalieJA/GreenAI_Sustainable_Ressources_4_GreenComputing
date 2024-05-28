#INSTALLATION DES BIBLIOTHEQUES
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# RECOLTE DES DONNEES
# Chargement des données MNIST (ensemble d'entraînement et de test)
mnist = tf.keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # image en entrée (x) et classe associée en sortie (y)

# Normalisation des données
x_train, x_test = x_train / 255.0, x_test / 255.0

# "Reshape" des données pour les rendre compatibles avec les CNNs
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# CREATION DU RESEAU DE NEURONES CONVOLUTIONNEL
model = tf.keras.models.Sequential([
    # Première couche de convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), #fonction d'activation choisie : relu
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Deuxième couche de convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Troisième couche de convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Couche Flatten pour convertir les matrices 2D en vecteurs 1D
    tf.keras.layers.Flatten(),
    
    # Couches entièrement connectées
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # Dernière couche avec softmax pour la classification des 10 classes
])

# ENTRAINEMENT DU MODELE SUR LES DONNEES D'ENTRAINEMENT
# Compilation du modèle avec une fonction de perte, un optimiseur et une métrique
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5)  # epochs = nb d'itérations

# EVALUATION DU MODELE SUR LES DONNEES DE TEST
test_loss, test_acc = model.evaluate(x_test, y_test)
print('\nTest accuracy:', test_acc)

# PREDICTION
# Chargement et pré-traitement des nouvelles images
files = ["zero.png", "one.png", "two.png", "three.png", "four.png", "five.png", "six.png", "seven.png"]
for file in files:
    img = load_img("in/" + file, color_mode='grayscale')
    x = img_to_array(img)
    x = x.reshape(1, 28, 28, 1)
    x = x / 255.0
    # Prédiction de la classe
    y = model.predict(x)
    y = y[0]
    bestClass = y.argmax()
    print(file + " => " + str(bestClass) + "\n")
