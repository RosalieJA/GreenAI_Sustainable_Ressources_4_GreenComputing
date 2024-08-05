# INSTALLATION DES BIBLIOTHEQUES
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
from codecarbon import EmissionsTracker

# RECOLTE DES DONNEES
# chargement des données MNIST (ensemble d'entraînement et de test)
mnist = tf.keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data() # image en entrée (x) et classe associée en sortie (y)

# Normalisation des données
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape des données pour GRU
x_train = x_train.reshape(-1, 28, 28)  # GRU attend un input de forme (batch_size, time_steps, input_dim)
x_test = x_test.reshape(-1, 28, 28)

# CREATION DU RESEAU GRU
model = tf.keras.models.Sequential([
    # Couche GRU : 2 couches cachées
    tf.keras.layers.GRU(128, input_shape=(28, 28), return_sequences=True),
    tf.keras.layers.GRU(128),
    # dernière couche de 10 neurones, entièrement connectée à la couche précédente
    tf.keras.layers.Dense(10, activation='softmax')
])

# ENTRAINEMENT DU MODELE SUR LES DONNEES D'ENTRAINEMENT
# Compilation du modèle avec une fonction de perte, un optimiseur et une métrique
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Suivi des émissions
tracker = EmissionsTracker()
tracker.start()

# Entraînement du modèle sur les données d'entraînement
print("Début de l'entraînement")
history = model.fit(x_train, y_train, epochs=5)  # epochs = nb d'itérations
print("Fin de l'entraînement")

# Fin du suivi des émissions
emissions = tracker.stop()
print(f"Emissions: {emissions} kg CO2")

# EVALUATION DU MODELE SUR LES DONNEES DE TEST
test_loss, test_acc = model.evaluate(x_test, y_test)
print('\nTest accuracy:', test_acc)

# PREDICTION
# Chargement et pré-traitement
files = ["zero.png", "one.png", "two.png", "three.png", "four.png", "five.png", "six.png", "seven.png"]
for file in files:
    img = load_img("in/" + file, color_mode='grayscale', target_size=(28, 28))
    x = img_to_array(img)
    x = x.reshape(1, 28, 28)
    x = x / 255.0

    y = model.predict(x)
    y = y[0]
    bestClass = y.argmax()
    print(file + " => " + str(bestClass) + "\n")
