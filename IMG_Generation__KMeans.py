#INSTALLATION DES BIBLIOTHEQUES
import numpy as np
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker
import time

# Dimensions de l'image
rows, cols = 256, 256

# Création de l'image de référence avec des cercles
image_ref = np.zeros((rows, cols))

# Dessiner quelques cercles
for x in range(rows):
    for y in range(cols):
        if (x - 128)**2 + (y - 128)**2 < 30**2 or (x - 64)**2 + (y - 64)**2 < 20**2 or (x - 192)**2 + (y - 192)**2 < 20**2:
            image_ref[x, y] = 1

# Affichage de l'image de référence
plt.imshow(image_ref, cmap='gray')
plt.title('Image de référence')
plt.axis('off')
plt.show()

from sklearn.cluster import KMeans

tracker = EmissionsTracker()
tracker.start()

# Redimensionner l'image de référence pour K-Means
X = image_ref.reshape(-1, 1)

# Appliquer K-Means pour regrouper les pixels en 2 clusters
start_time = time.time()
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
end_time = time.time()

generation_time = end_time - start_time

# Attribuer des couleurs aux clusters
image_kmeans = kmeans.labels_.reshape((rows, cols))

emissions = tracker.stop()

# Affichage de l'image générée avec K-Means
plt.imshow(image_kmeans, cmap='gray')
plt.title('Image générée avec K-Means')
plt.axis('off')
plt.show()

print(f"Temps de génération pour K-Means: {generation_time} secondes")
print(f"Emissions for K-Means: {emissions} kg CO2")
