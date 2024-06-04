#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace std;

// Fonction sigmoïde
double sigmoide(double x) {
    return 1 / (1 + exp(-x));
}

// Dérivée de la fonction sigmoïde
double derivee_sigmoide(double x) {
    return x * (1 - x);
}

// Fonction ReLU
double relu(double x) {
    return max(0.0, x);
}

// Dérivée de la fonction ReLU
double derivee_relu(double x) {
    return x > 0 ? 1.0 : 0.0;
}

// Addition de matrices
vector<vector<double>> matrix_add(const vector<vector<double>>& a, const vector<vector<double>>& b) {
    vector<vector<double>> result(a.size(), vector<double>(a[0].size()));
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a[0].size(); ++j) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    return result;
}

// Multiplication élémentaire de matrices
vector<vector<double>> elementwise_multiply(const vector<vector<double>>& a, const vector<vector<double>>& b) {
    vector<vector<double>> result(a.size(), vector<double>(a[0].size()));
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a[0].size(); ++j) {
            result[i][j] = a[i][j] * b[i][j];
        }
    }
    return result;
}

// Classe du réseau multi-couches
class ReseauMultiCouches {
public:
    int nombre_couches_cachees;
    vector<vector<vector<double>>> poids;
    vector<vector<double>> biais;
    vector<vector<vector<double>>> sorties;

    ReseauMultiCouches(int taille_entree, int taille_cachee, int taille_sortie, int nombre_couches_cachees = 10) {
        this->nombre_couches_cachees = nombre_couches_cachees;
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> d(0, 1);

        // Initialisation des poids et biais pour les couches cachées
        poids.push_back(init_matrix(taille_entree, taille_cachee, gen, d, sqrt(2.0 / taille_entree)));
        biais.push_back(vector<double>(taille_cachee, 0.0));

        // Couches cachées intermédiaires
        for (int i = 1; i < nombre_couches_cachees; ++i) {
            poids.push_back(init_matrix(taille_cachee, taille_cachee, gen, d, sqrt(2.0 / taille_cachee)));
            biais.push_back(vector<double>(taille_cachee, 0.0));
        }

        // Couche de sortie
        poids.push_back(init_matrix(taille_cachee, taille_sortie, gen, d, sqrt(2.0 / taille_cachee)));
        biais.push_back(vector<double>(taille_sortie, 0.0));
    }

    vector<vector<double>> propagation_avant(const vector<vector<double>>& x) {
        sorties.clear();
        vector<vector<double>> sortie = x;
        sorties.push_back(sortie);

        // Propagation à travers les couches cachées avec ReLU
        for (int i = 0; i < nombre_couches_cachees; ++i) {
            sortie = activation(sortie, poids[i], biais[i], relu);
            sorties.push_back(sortie);
        }

        // Propagation à travers la couche de sortie avec Sigmoïde
        sortie = activation(sortie, poids.back(), biais.back(), sigmoide);
        sorties.push_back(sortie);

        return sortie;
    }

    void retropropagation(const vector<vector<double>>& x, const vector<vector<double>>& y, double taux_apprentissage) {
        vector<vector<double>> erreur = matrix_subtract(y, sorties.back());
        vector<vector<double>> d_sortie = elementwise_multiply(erreur, matrix_apply(sorties.back(), derivee_sigmoide));

        poids.back() = matrix_add(poids.back(), matrix_multiply(transpose(sorties[nombre_couches_cachees]), d_sortie, taux_apprentissage));
        biais.back() = vector_add(biais.back(), sum_columns(d_sortie, taux_apprentissage));

        vector<vector<double>> d_cachee = d_sortie;
        for (int i = nombre_couches_cachees - 1; i >= 0; --i) {
            d_cachee = elementwise_multiply(matrix_multiply(d_cachee, transpose(poids[i + 1])), matrix_apply(sorties[i + 1], derivee_relu));
            poids[i] = matrix_add(poids[i], matrix_multiply(transpose(sorties[i]), d_cachee, taux_apprentissage));
            biais[i] = vector_add(biais[i], sum_columns(d_cachee, taux_apprentissage));
        }
    }

    void entrainer(const vector<vector<double>>& x, const vector<vector<double>>& y, int epochs, double taux_apprentissage) {
        for (int i = 0; i < epochs; ++i) {
            propagation_avant(x);
            retropropagation(x, y, taux_apprentissage);
            if (i % 1000 == 0) {
                double perte = mean_squared_error(y, sorties.back());
                cout << "Epoque " << i << ", Perte: " << perte << endl;
            }
        }
    }

    double tester(const vector<vector<double>>& x, const vector<vector<double>>& y) {
        vector<vector<double>> pred = propagation_avant(x);
        vector<vector<double>> resultat = matrix_equal(pred, y);
        return mean(resultat);
    }

private:
    vector<vector<double>> activation(const vector<vector<double>>& input, const vector<vector<double>>& poids, const vector<double>& biais, double(*activation_func)(double)) {
        vector<vector<double>> result(input.size(), vector<double>(poids[0].size()));
        for (size_t i = 0; i < input.size(); ++i) {
            for (size_t j = 0; j < poids[0].size(); ++j) {
                result[i][j] = biais[j];
                for (size_t k = 0; k < input[0].size(); ++k) {
                    result[i][j] += input[i][k] * poids[k][j];
                }
                result[i][j] = activation_func(result[i][j]);
            }
        }
        return result;
    }

    vector<vector<double>> init_matrix(int rows, int cols, mt19937& gen, normal_distribution<>& d, double scale) {
        vector<vector<double>> matrix(rows, vector<double>(cols));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix[i][j] = d(gen) * scale;
            }
        }
        return matrix;
    }

    vector<vector<double>> transpose(const vector<vector<double>>& matrix) {
        vector<vector<double>> result(matrix[0].size(), vector<double>(matrix.size()));
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[0].size(); ++j) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    vector<vector<double>> matrix_multiply(const vector<vector<double>>& a, const vector<vector<double>>& b, double scale = 1.0) {
        vector<vector<double>> result(a.size(), vector<double>(b[0].size()));
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < b[0].size(); ++j) {
                for (size_t k = 0; k < b.size(); ++k) {
                    result[i][j] += a[i][k] * b[k][j];
                }
                result[i][j] *= scale;
            }
        }
        return result;
    }

    vector<double> vector_add(const vector<double>& a, const vector<double>& b) {
        vector<double> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    vector<vector<double>> matrix_subtract(const vector<vector<double>>& a, const vector<vector<double>>& b) {
        vector<vector<double>> result(a.size(), vector<double>(a[0].size()));
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < a[0].size(); ++j) {
                result[i][j] = a[i][j] - b[i][j];
            }
        }
        return result;
    }

    vector<vector<double>> matrix_apply(const vector<vector<double>>& matrix, double(*func)(double)) {
        vector<vector<double>> result(matrix.size(), vector<double>(matrix[0].size()));
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[0].size(); ++j) {
                result[i][j] = func(matrix[i][j]);
            }
        }
        return result;
    }

    vector<double> sum_columns(const vector<vector<double>>& matrix, double scale = 1.0) {
        vector<double> result(matrix[0].size(), 0.0);
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[0].size(); ++j) {
                result[j] += matrix[i][j] * scale;
            }
        }
        return result;
    }

    double mean_squared_error(const vector<vector<double>>& a, const vector<vector<double>>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < a[0].size(); ++j) {
                sum += pow(a[i][j] - b[i][j], 2);
            }
        }
        return sum / (a.size() * a[0].size());
    }

    double mean(const vector<vector<double>>& matrix) {
        double sum = 0.0;
        for (const auto& row : matrix) {
            for (double value : row) {
                sum += value;
            }
        }
        return sum / (matrix.size() * matrix[0].size());
    }

    vector<vector<double>> matrix_equal(const vector<vector<double>>& a, const vector<vector<double>>& b) {
        vector<vector<double>> result(a.size(), vector<double>(a[0].size()));
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < a[0].size(); ++j) {
                result[i][j] = (a[i][j] == b[i][j]) ? 1.0 : 0.0;
            }
        }
        return result;
    }

    void print_matrix(const vector<vector<double>>& matrix, const string& name) {
        cout << name << ":" << endl;
        for (const auto& row : matrix) {
            for (double value : row) {
                cout << value << " ";
            }
            cout << endl;
        }
    }

    void print_vector(const vector<double>& vec, const string& name) {
        cout << name << ": ";
        for (double value : vec) {
            cout << value << " ";
        }
        cout << endl;
    }
};

// TEST
int main() {
    // Données d'entrée
    vector<vector<double>> x = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    // Données de sortie
    vector<vector<double>> y = {
        {0},
        {1},
        {1},
        {0}
    };

    // Création d'un réseau multi-couches avec 10 couches cachées
    ReseauMultiCouches reseau(2, 5, 1, 10);

    // Entraînement du réseau
    reseau.entrainer(x, y, 10000, 0.01);

    // Test du réseau
    cout << "Donnees en entree :" << endl;
    for (const auto& row : x) {
        for (double value : row) {
            cout << value << " ";
        }
        cout << endl;
    }

    cout << "Sorties apres l'entrainement:" << endl;
    vector<vector<double>> sorties = reseau.propagation_avant(x);
    for (const auto& row : sorties) {
        for (double value : row) {
            cout << value << " ";
        }
        cout << endl;
    }

    // Calcul et affichage de l'exactitude
    double exactitude = reseau.tester(x, y);
    cout << "Exactitude du reseau: " << exactitude * 100 << "%" << endl;

    return 0;
}

