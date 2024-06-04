# TX00_GREEN_AI

BILAN DES DIFFERENTS CODES ET RESULTATS :  

SANS LIBRAIRIES => utilisation de RAPL pour suivre les émissions
- Exemple de Machine Learning (test avec XOR) : A venir
- Exemple de Deep Learning (test avec XOR) :
   - Python : perceptron multi-couches (10 couches cachées) => tests à venir
   -  C++ : perceptron multi-couches (10 couches cachées) => tests à venir


AVEC UTILISATION DE LIBRAIRIES : Tensorflow, scikit learn, etc. => utilisation de CodeCarbon pour suivre les émissions (cf. https://codecarbon.io/)

Deep Learning : exemples avec MNIST
- Exemple de ANN : perceptron multi-couches (10 couches cachées) => Emissions: 1.4007240778055813e-05 kg CO2 et Test accuracy: 0.9797999858856201
- Exemple de CNN (7 couches cachées) => Emissions: 3.7664445672275096e-05 kg CO2 et Test accuracy: 0.9904999732971191
- Exemple de RNN : GRU (2 couches cachées) => Emissions: 0.00010874017996317586 kg CO2 et Test accuracy: 0.9869999885559082
- Exemple de GAN (5 couches cachées dans générateur + 5 couches cachées dans discriminateur) => A venir 

Machine Learning : exemples avec MNIST
- Supervisé : KNN (3 voisins) => Émissions : 9.00082625136501e-06 kg CO2 et Test accuracy: 0.9447
- Non supervisé : KMEAN (256 clusters) => Emissions : 1.3420924768623704e-06 kg CO2 et Test accuracy: 0.9014

