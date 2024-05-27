#INSTALLATION DES BIBLIOTHEQUES
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

#RECOLTE DES DONNEES
#chargement des données MNIST (ensemble d'entraînement et de test)
mnist = tf.keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data() #image en entrée (x) et classe associée en sortie (y)

# Normalisation des données
x_train, x_test = x_train / 255.0, x_test / 255.0

#CREATION DU RESEAU DE NEURONE
model = tf.keras.models.Sequential([
    # première couche pour recevoir tous les pixels en entrée : 
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    #couches cachées : ATTENTION TROP DE COUCHES CACHEES PEUT MENER AU SURAPPRENTISSAGE
    tf.keras.layers.Dense(128, activation='relu'), #tanh et sigmoid moins bien que relu ? 
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    # dernière couche de 10 neurones, entièrement connectée à la couche précédente
    tf.keras.layers.Dense(10),
    # traitement pour ramener toutes les valeurs entre 0 et 1
    tf.keras.layers.Softmax() 
])

#ENTRAINEMENT DU MODELE SUR LES DONNEES D'ENTRAINEMENT
# Compilation du modèle avec une fonction de perte, un optimiseur et une métrique
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history=model.fit(x_train, y_train, epochs=5) #epochs = nb d'itérations

#EVALUATION DU MODELE SUR LES DONNEES DE TEST
test_loss, test_acc = model.evaluate(x_test, y_test)
print('\nTest accuracy:', test_acc)

#PREDICTION
#Chargement et pré-traitement
files = ["zero.png", "one.png", "two.png", "three.png", "four.png", "five.png", "six.png", "seven.png"]
for file in files:
    img = load_img("in/" + file, color_mode='grayscale')
    x = img_to_array(img)
    x = x.reshape(1,28,28)
    x = x / 255.0
    #Prédiction de la classe       
    y = model.predict(x)
    y = y[0]
    bestClass = y.argmax()
    print(file + " => " + str(bestClass) + "\n")
