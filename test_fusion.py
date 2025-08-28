import numpy as np
import methodes_fusion as fs
from matplotlib import pyplot as plt
from importlib import reload
import pickle
reload(fs)
from tensorflow.keras.models import load_model
from metric_seq2seq import AUCSeq, F1ScoreSeq, PrecisionSeq, RecallSeq
import pickle
import os

custom_objects = {
    "AUCSeq": AUCSeq,
    "F1ScoreSeq": F1ScoreSeq,
    "PrecisionSeq": PrecisionSeq,
    "RecallSeq": RecallSeq
}

micros = [1,2,3,4,12,1234]
accs = []
mics_fus = []
accs_fus = []
train_models_probas = []
y_models_probas = []
#train probas 
# Données d'entraînement pour la fusion (calibration)
for mic in micros: 
    path_fichier_poids = os.path.join("models","audios", f"audio_{mic}.pkl")
    with open(path_fichier_poids, "rb") as f:
        data = pickle.load(f)
        train_models_probas.append(data[1][1])
        y_true = data[2][0].argmax(axis=1).flatten()
        y_models_probas.append(data[2][1])

for acc in accs : 
    path_fichier_poids = os.path.join("models","acceleros", f"acc_{acc}.pkl")
    with open(path_fichier_poids, "rb") as f:
        data = pickle.load(f)
        train_models_probas.append(data[1][1])
        y_true = data[2][0].argmax(axis=1).flatten()
        y_models_probas.append(data[2][1])

for acc,mic in zip(accs_fus,mics_fus) : 
    path_fichier_poids = os.path.join("models","fusion", f"acc{acc}_aud{mic}.pkl")
    with open(path_fichier_poids, "rb") as f:
        data = pickle.load(f)
        train_models_probas.append(data[1][1])
        y_true = data[2][0].argmax(axis=1).flatten()
        y_models_probas.append(data[2][1])


# Convertir en tenseur (n_models, n_samples, n_classes)

train_predictions = np.array(train_models_probas)




# Convertir en tenseur (n_models, n_samples, n_classes)


predictions = np.array(y_models_probas)


print("Shape of predictions:", predictions.shape)  # Doit être (4, 10, 2)
from sklearn.metrics import recall_score,f1_score,precision_score
# Appel de ta fonction
methodes = ["majority_vote","borda_count","dempster_shafer","locality_based"]   #"bks","logistic_regression",
scores = []
yfs = []
for meeth in methodes : 
    yfs.append(fs.fusion_prediction_ensemble(predictions=predictions, true_labels=y_true, method=meeth,train_predictions=train_predictions,train_true_labels=y_true))

for i,mic in enumerate(micros):
    yp = f1_score(y_true=y_true,y_pred=(y_models_probas[i][:,1] > 0.5).astype(int),average="binary",pos_label=1)
    scores.append(yp)
    print(f" score du micro {mic} {yp}" )

for i,acc in enumerate(accs):
    
    yp = f1_score(y_true=y_true,y_pred=(y_models_probas[i+len(micros)][:,1] > 0.5).astype(int),average="binary",pos_label=1)
    scores.append(yp)
    print(f" score d'accelero {acc} {yp}" )

for i,(acc,mic) in enumerate(zip(accs_fus,mics_fus)):
    
    yp = f1_score(y_true=y_true,y_pred=(y_models_probas[i+len(micros) +len(accs)][:,1] > 0.5).astype(int),average="binary",pos_label=1)
    scores.append(yp)
    print(f" score de la fusion {"acc"+str(acc)+"_aud"+str(mic)} {yp}" )

for i,meth in enumerate(methodes) : 
    yf = f1_score(y_true=y_true,y_pred=yfs[i],average="binary",pos_label=1)
    if meth == "majority_vote" : 
        y_fusion = yfs[i]
    scores.append(yf)
    print(f"score de la methode {meth} est {yf}")



fig, ax = plt.subplots()

hbars = ax.barh(["micro"+str(mic) for mic in micros] +["acc"+str(acc) for acc in accs]+["acc"+str(acc)+"_"+"aud"+str(aud) for acc,aud in zip(accs_fus,mics_fus)] +methodes, scores, align='center')
#plt.barh([str(mic) for mic in micros] + methodes, scores)

ax.bar_label(hbars, fmt='%.4f')

#afficher la matrice de confusion avec la meilleur methode
sets = ["Train", "Val", "Test"]
plt.figure(figsize=(15,8))

import seaborn as sns
from sklearn.metrics import confusion_matrix
if True:
    cm = confusion_matrix(y_true,y_fusion)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Création des annotations : "xx.xx%\n(nb)"
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{cm_normalized[i, j]*100:.2f}%\n({cm[i, j]})"

    sns.heatmap(cm_normalized, annot=annotations, fmt="", cmap="Blues")
    plt.title(f"{sets[2]} Confusion Matrix des données de test")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    


plt.show()