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

tracker = EmissionsTracker()
tracker.start()

# Calcul de la transformée de Fourier de l'image de référence
frequencies = np.fft.fft2(image_ref)

# Inverse de la transformée de Fourier pour obtenir l'image
start_time = time.time()
image_fft = np.fft.ifft2(frequencies).real
end_time = time.time()

generation_time = end_time - start_time

emissions = tracker.stop()

# Affichage de l'image générée avec la Transformée de Fourier
plt.imshow(image_fft, cmap='gray')
plt.title('Image générée avec la Transformée de Fourier')
plt.axis('off')
plt.show()

print(f"Temps de génération pour la Transformée de Fourier: {generation_time} secondes")
print(f"Emissions for Fourier transform: {emissions} kg CO2")
