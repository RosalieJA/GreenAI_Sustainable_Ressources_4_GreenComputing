BILAN DES DIFFERENTS CODES ET RESULTATS :


SANS LIBRAIRIES => utilisation de RAPL pour suivre les émissions pour comparer les langages

- Exemple de Deep Learning (test avec XOR) :
    - Python : perceptron multi-couches (10 couches cachées)
    - C++ : perceptron multi-couches (10 couches cachées)
 - Conclusion : C++ consomme moins que Python (confirme littérature)


AVEC UTILISATION DE LIBRAIRIES : Tensorflow, scikit learn, etc. => utilisation de CodeCarbon pour suivre les émissions https://codecarbon.io/

- Machine Learning : exemples avec MNIST
    - Supervisé : KNN (3 voisins) => Émissions : 9.00082625136501e-06 kg CO2 et Test accuracy: 0.9447
    - Non supervisé : KMEAN (256 clusters) => Emissions : 1.3420924768623704e-06 kg CO2 et Test accuracy: 0.9014

- Deep Learning : exemples avec MNIST
    - Exemple de ANN : perceptron multi-couches (10 couches cachées) => Emissions: 1.4007240778055813e-05 kg CO2 et Test accuracy: 0.9797999858856201
    - Exemple de CNN (7 couches cachées) => Emissions: 3.7664445672275096e-05 kg CO2 et Test accuracy: 0.9904999732971191
    - Exemple de RNN : GRU (2 couches cachées) => Emissions: 0.00010874017996317586 kg CO2 et Test accuracy: 0.9869999885559082
    - Exemple de GAN, à l'aide de la documentation officielle Tensorflow (5 couches cachées dans générateur + 5 couches cachées dans discriminateur) => ATTENTION : ne peux pas être run sur n'importe quel ordi


NOTE BENE : L'exemple de Transformeur (comme dans ChatGPT) "Attention is all you need" (à l'aide de la documentation officielle Tensorflow) n'est pas terminé (problèmes lors du débbugage) => dans la branche Essais => IL NE FAUT PAS MERGE
  


