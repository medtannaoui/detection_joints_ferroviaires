import numpy as np
from collections import Counter
from typing import List, Optional
from sklearn.linear_model import LogisticRegression
class BehaviorKnowledgeSpace:
    """
    Implémentation de la méthode BKS (Behavior Knowledge Space) pour la fusion de classificateurs.
    Elle utilise une table de correspondance entre combinaisons de prédictions et étiquettes réelles.
    """

    def __init__(self, precision=2):
        self.precision = precision
        self.table = {}
        self.known_keys = []

    def fit(self, predictions, true_labels):
        """
        Entraîne la table BKS sur les prédictions des classificateurs et les labels réels.
        """
        self.table = {}
        self.known_keys = []

        for pred_vec, label in zip(predictions, true_labels):
            key = tuple(np.round(np.array(pred_vec).flatten(), self.precision))
            print(type(label),label)
            if key not in self.table:
                self.table[key] = {}
                self.known_keys.append(key)
            self.table[key][label] = self.table[key].get(label, 0) + 1

    def predict(self, predictions):
        """
        Prédit les classes en utilisant la table BKS. Si une combinaison est inconnue,
        utilise la clé la plus proche en distance de Hamming.
        """
        result = []
        for pred_vec in predictions:
            key = tuple(np.round(np.array(pred_vec).flatten(), self.precision))
            if key in self.table:
                label_counts = self.table[key]
            else:
                # Clé inconnue → Cherche la clé la plus proche (distance de Hamming)
                best_key = min(
                    self.known_keys,
                    key=lambda known: sum(a != b for a, b in zip(known, key))
                )
                label_counts = self.table[best_key]

            # Choisir le label majoritaire
            best_label = max(label_counts.items(), key=lambda x: x[1])[0]
            result.append(best_label)

        return result


def convert_predictions_to_mass_functions(predictions, alpha=1.0):
    """
    Convertit les prédictions en fonctions de masse pour la théorie des croyances.
    Supporte le binaire et le multi-classes.
    """
    predictions = np.array(predictions)
    n_models, n_samples, n_classes = predictions.shape

    uniform_preds = []

    for m in range(n_models):
        model_preds = []
        for s in range(n_samples):
            pred = predictions[m, s]
            if n_classes == 1 or isinstance(pred, (float, np.floating, int)):
                prob_1 = float(pred[0]) if isinstance(pred, (list, np.ndarray)) else float(pred)
                prob_0 = 1.0 - prob_1
                model_preds.append([prob_0 * alpha, prob_1 * alpha])
            else:
                norm = np.array(pred) / np.sum(pred)
                model_preds.append(norm * alpha)
        uniform_preds.append(model_preds)

    return list(zip(*uniform_preds))  # (n_samples, n_models, n_classes)


def combine_mass_dempster_rule(m1, m2):
    """
    Combine deux fonctions de masse en utilisant la règle de Dempster.
    """
    combined = {}
    conflict = 0.0
    for A in m1:
        for B in m2:
            intersection = tuple(sorted(set(A).intersection(B)))
            if intersection:
                combined[intersection] = combined.get(intersection, 0) + m1[A] * m2[B]
            else:
                conflict += m1[A] * m2[B]

    if conflict < 1:
        for k in combined:
            combined[k] /= (1 - conflict)
    else:
        for k in combined:
            combined[k] = 0
    return combined


def fusion_prediction_ensemble(predictions, method="majority_vote", alpha=0.9, 
                                true_labels=None, train_predictions=None, train_true_labels=None):
    """
    Fusionne les prédictions de plusieurs classificateurs selon différentes méthodes.

    Args :
        predictions : liste de prédictions des modèles (labels ou probabilités).
        method : méthode de fusion ('majority_vote', 'borda_count', 'dempster_shafer', etc.).
        alpha : paramètre de confiance pour certaines méthodes.
        true_labels : labels réels (si nécessaire pour certaines méthodes).
        train_predictions : pour BKS ou autres méthodes à apprentissage sur les prédictions.
        train_true_labels : pour BKS ou autres méthodes à apprentissage sur les prédictions.

    Retour :
        Liste des prédictions fusionnées.
    """
    predictions = np.array(predictions)
    n_models, n_samples, n_classes = predictions.shape

    if method == "majority_vote":
        threshold = 0.5
        results = []
        for i in range(n_samples):
            votes = []
            for m in range(n_models):
                pred = predictions[m, i]
                if n_classes == 1:
                    votes.append(1 if pred[0] >= threshold else 0)
                else:
                    votes.append(np.argmax(pred))
            count = Counter(votes)
            most_common = count.most_common()
            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:         #égalité
                results.append(1)  # Egalité → classe 1
            else:
                results.append(most_common[0][0])
        return results

    elif method == "borda_count":
        per_sample = convert_predictions_to_mass_functions(predictions, alpha=1)
        results = []
        for preds in per_sample:
            scores = np.zeros(n_classes)
            for pred in preds:
                ranked = np.argsort(pred)[::-1]
                for rank, cls in enumerate(ranked):
                    scores[cls] += n_classes - rank - 1
            results.append(int(np.argmax(scores)))
        return results

    elif method == "dempster_shafer":
        per_sample = convert_predictions_to_mass_functions(predictions, alpha=alpha)
        results = []
        for preds in per_sample:
            frame = tuple(range(n_classes))
            mass_list = []
            for pred in preds:
                norm = np.array(pred) / np.sum(pred)
                mass = {(i,): norm[i] for i in range(n_classes)}
                mass[frame] = 1 - alpha
                for key in mass:
                    mass[key] *= alpha
                mass_list.append(mass)

            combined_mass = mass_list[0]
            for m in mass_list[1:]:
                combined_mass = combine_mass_dempster_rule(combined_mass, m)

            if (0,) in combined_mass and combined_mass[(0,)] < 0.7:
                best_cls = max(
                    (cls for cls in combined_mass if len(cls) == 1),
                    key=lambda x: combined_mass[x],
                    default=(0,)
                )
                results.append(best_cls[0])
            else:
                results.append(0)
        return results

    elif method == "bks":
        rounded_train_preds = np.round(train_predictions, 1)  # shape : (n_models, n_samples, n_classes)
        rounded_test_preds = np.round(predictions, 1)

        bks_table = {}

        n_models, n_train_samples, n_classes = rounded_train_preds.shape

        # Construction du dico BKS
        for i in range(n_train_samples):
            key = tuple(rounded_train_preds[:, i, :].flatten())
            if key not in bks_table:
                bks_table[key] = []
            bks_table[key].append(train_true_labels[i])

        fused_labels = []

        # Application sur les prédictions test
        n_test_samples = rounded_test_preds.shape[1]

        for i in range(n_test_samples):
            key = tuple(rounded_test_preds[:, i, :].flatten())

            if key in bks_table:
                labels_observed = bks_table[key]

                if 1 in labels_observed:
                    fused_labels.append(1)
                else:
                    # Prendre la classe majoritaire (dans ce cas ce sera 0, mais on garde la logique générale)
                    majority_label = int(np.round(np.mean(labels_observed)))
                    fused_labels.append(majority_label)
            else:
                # Si la combinaison n'existe pas, prédire la classe majoritaire globale du train
                global_majority = int(np.round(np.mean(train_true_labels)))
                fused_labels.append(global_majority)

        return fused_labels

    

    elif method == "logistic_regression":
        if train_predictions is None or train_true_labels is None:
            raise ValueError("La régression logistique nécessite les prédictions d'entraînement et les labels.")

        # Reshape des données d'entraînement : (n_samples_train, n_models * 1)
        train_predictions = np.array(train_predictions)
        n_models_train, n_samples_train, n_classes_train = train_predictions.shape
        X_train = train_predictions.transpose(1, 0, 2)[:,:,1].reshape(n_samples_train, -1)
        

        # Entraînement du modèle
        clf = LogisticRegression(max_iter=1000,class_weight='balanced')
        clf.fit(X_train, train_true_labels)

        # Reshape des données de test : (n_samples_test, n_models * 1)
        test_predictions = predictions.transpose(1, 0, 2)[:,:,1].reshape(n_samples, -1)

        # Prédictions
        pred_labels = clf.predict(test_predictions)
        return pred_labels
    
    elif method == "locality_based":
        from sklearn.neighbors import NearestNeighbors

        if train_predictions is None or train_true_labels is None:
            raise ValueError("La méthode 'locality_based' nécessite 'train_predictions' et 'train_true_labels'.")

        preds = np.array(train_predictions)  # (n_models, n_train_samples, n_classes)
        test_preds = np.array(predictions)   # (n_models, n_test_samples, n_classes)

        n_models, n_train_samples, n_classes = preds.shape
        _, n_test_samples, _ = test_preds.shape
        k = 3  # nombre de voisins

        label_train = np.array(train_true_labels)

        # Gestion du binaire : s'assurer qu'on a toujours (n_classes=2) pour éviter les problèmes avec argmax
        if n_classes == 1:
            preds = np.concatenate([1 - preds, preds], axis=-1)
            test_preds = np.concatenate([1 - test_preds, test_preds], axis=-1)
            n_classes = 2

        # Vectorisation des probas : concaténation des modèles pour chaque sample
        X_train = preds.transpose(1, 0, 2).reshape(n_train_samples, -1)  # (n_train_samples, n_models * n_classes)

        knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        knn.fit(X_train)

        fused_labels = []

        for i in range(n_test_samples):
            x_i = test_preds[:, i, :].reshape(1, -1)  # (1, n_models * n_classes)

            _, idxs = knn.kneighbors(x_i)
            idxs = idxs[0]

            model_scores = []

            for m in range(n_models):
                pred_labels_neighbors = np.argmax(preds[m][idxs], axis=1)
                true_labels_neighbors = label_train[idxs]

                acc_local = np.mean(pred_labels_neighbors == true_labels_neighbors)
                model_scores.append(acc_local)

            # Choix du modèle localement le plus fiable
            best_model_idx = np.argmax(model_scores)
            best_prediction = np.argmax(test_preds[best_model_idx, i])
            fused_labels.append(best_prediction)

        return fused_labels

    

    else:
        raise ValueError(f"Méthode de fusion non supportée : {method}")
    

    

        


