#INSTALLATION DE LA BIBLIOTHEQUE
import numpy as np

#CREATION DES FONCTIONS ET CLASSE DE BASE
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def derivee_sigmoide(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def derivee_relu(x):
    return np.where(x > 0, 1, 0)

class ReseauMultiCouches:
    def __init__(self, taille_entree, taille_cachee, taille_sortie, nombre_couches_cachees=10):
        self.nombre_couches_cachees = nombre_couches_cachees
        
        # Initialisation des poids et biais pour les couches cachées
        self.poids = []
        self.biais = []
        
        # Poids et biais pour la première couche cachée
        self.poids.append(np.random.randn(taille_entree, taille_cachee) * np.sqrt(2. / taille_entree))
        self.biais.append(np.zeros(taille_cachee))
        
        # Poids et biais pour les couches cachées intermédiaires
        for _ in range(1, nombre_couches_cachees):
            self.poids.append(np.random.randn(taille_cachee, taille_cachee) * np.sqrt(2. / taille_cachee))
            self.biais.append(np.zeros(taille_cachee))
        
        # Poids et biais pour la couche de sortie
        self.poids.append(np.random.randn(taille_cachee, taille_sortie) * np.sqrt(2. / taille_cachee))
        self.biais.append(np.zeros(taille_sortie))

    #FONCTION DE PROPAGATION AVANT
    def propagation_avant(self, x):
        self.sorties = [x]
        
        # Propagation à travers les couches cachées avec ReLU
        for i in range(self.nombre_couches_cachees):
            x = relu(np.dot(x, self.poids[i]) + self.biais[i])
            self.sorties.append(x)
        
        # Propagation à travers la couche de sortie avec Sigmoïde
        x = sigmoide(np.dot(x, self.poids[-1]) + self.biais[-1])
        self.sorties.append(x)
        
        return x

    #FONCTION DE RETRO-PROPOPAGATION
    def retropropagation(self, x, y, taux_apprentissage):
        # Calcul de l'erreur de sortie
        erreur = y - self.sorties[-1]
        d_sortie = erreur * derivee_sigmoide(self.sorties[-1])
        
        # Gradients pour la couche de sortie
        self.poids[-1] += np.dot(self.sorties[-2].T, d_sortie) * taux_apprentissage
        self.biais[-1] += np.sum(d_sortie, axis=0) * taux_apprentissage
        
        # Gradients pour les couches cachées
        d_cachee = d_sortie
        for i in reversed(range(self.nombre_couches_cachees)):
            d_cachee = d_cachee.dot(self.poids[i + 1].T) * derivee_relu(self.sorties[i + 1])
            self.poids[i] += np.dot(self.sorties[i].T, d_cachee) * taux_apprentissage
            self.biais[i] += np.sum(d_cachee, axis=0) * taux_apprentissage

    #FONCTION D'ENTRAINEMENT DU MLP
    def entrainer(self, x, y, epoques, taux_apprentissage):
        for epoque in range(epoques):
            self.propagation_avant(x)
            self.retropropagation(x, y, taux_apprentissage)
            if epoque % 1000 == 0:
                perte = np.mean(np.square(y - self.sorties[-1]))
                print(f"Époque {epoque}, Perte: {perte}")

    #FONCTION POUR CALCULER L'ACCURACY
    def tester(self, x, y):
        # Propagation avant pour obtenir les prédictions
        predictions = self.propagation_avant(x)
        # Arrondir les prédictions pour obtenir des valeurs binaires
        predictions_arrondies = np.round(predictions)
        # Calculer le nombre de prédictions correctes
        exactitude = np.mean(predictions_arrondies == y)
        return exactitude

#UTILISATION : EXEMPLE DE LA PORTE XOR
#Données d'entrée
x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

#Données de sortie
y = np.array([[0],
              [1],
              [1],
              [0]])

# Création d'un réseau multi-couches avec 10 couches cachées
reseau = ReseauMultiCouches(taille_entree=2, taille_cachee=5, taille_sortie=1, nombre_couches_cachees=10)

# Entraînement du réseau
reseau.entrainer(x, y, epoques=10000, taux_apprentissage=0.01)

# Test du réseau
print("Données en entrée :")
print(x)

print("Sorties après l'entraînement:")
sorties = reseau.propagation_avant(x)
print(sorties)

# Calcul et affichage de l'exactitude
exactitude = reseau.tester(x, y)
print(f"Exactitude du réseau: {exactitude * 100:.2f}%")
