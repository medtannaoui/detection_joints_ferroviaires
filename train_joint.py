'''
Fcihier pour faire l'entrainemnet
'''
import random
random.seed(0)

import os 
import pickle
from importlib import reload
import functions as fct
reload(fct)

from tensorflow.keras.metrics import AUC,Precision,F1Score,Recall
from tensorflow.keras.losses import BinaryFocalCrossentropy,CategoricalFocalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import metric_seq2seq
reload(metric_seq2seq)
from metric_seq2seq import PrecisionSeq, RecallSeq, AUCSeq, F1ScoreSeq

ind_mf4_train = 4   #indice du mf4 utilisé
total = True  #if true on fusionne entre les audios et les accéléros
ind_mic = [1,2]    # listes des indices des mf4 utilisés
ind_acc = [3,4]    # listes des acceleros à utiliser
sr = 100000      # Fréquence d'echantillonage
tfn = 0.02       # Taille de la fenetre en secondes
FMAX = 10000      # Fréquence maximale en HZ seulement dans le cas sans fusion entre les deux modalités si non (22000 pour audio et 10khz pour l'acceéleros)
seq2seq = False  # Si true on va faire une approche trame à trame (labelisation par trames) si False labélisation par séquence

N_window = fct.get_near_pof2(tfn,sr)
NB_FEATURES = 64    # Nombre de coeficients mels utilisés
nbr_classes = 2     # Nombre de classes (si 2 il y aura "joint" et "pas de joint") si 3 (il y aura "roue_avant" et "roue_arrière" et "rien")
apply_pca = False   # Si True , la pca est appliqué avant la création des séquences

NbrFilterByLayer = [32,64]   # Liste des tailles de filtres utilisés pour les couches de convolution.
NbrNeurByReccurLayer = [32,16]  # Liste des Nombre de neuronnes par chaque couche réccurente.
neurByDense = []         # Liste des nombre de neuronnes pour les couches dense.
batch_size = 128              # La taille du batch utilisée lors de l'entrainmenet.
epochs = 150                  # Le nombre d'éteration utilisé pour l'entrainement.
initial_learning_rate = 5e-5  # Taux d'apprentissage initiale utilisé dans Expodecay.
feature = "acc"             # si "audio" on s'entraine sur les microphones , si "acc" on entraine sur les accélérométres.
    #extraire les coefs mels
if True:  
    mels ,labels = [],[]
    for debut,fin in zip([int(1*60)],[int(17*60 +40)]):
        if  not total:
            xi,yi = fct.mels_features(list_mf4 = [ind_mf4_train],
                                        list_mics = [1,2],
                                        N_window = N_window,
                                        nb_mels = NB_FEATURES,
                                        power = 2.0,
                                        debut = debut,
                                        fin = fin,
                                        annot = True,
                                        sous_mean = True,
                                        feature = "audio",
                                        f_max = FMAX,
                                        overlap = 2, #25% de hop_lenght pour le calule des mels
                                        energie = "both",
                                        precoce = True,
                                        derive1 =True,
                                        derive2 = True,
                                        
                                        nbr_classes= 2 )
        
            
        elif total :
            mels ,labels = [],[] 
            for debut,fin in zip([int(1*60)],[int(17*60 +40)]):    #zip([ind_mf4_train],[int(1*60)],[int(17*60+40)])
                xi,yi = fct.mels_features(list_mf4 = [ind_mf4_train],
                                            list_mics = ind_mic,
                                            N_window = N_window,
                                            debut = debut,
                                            fin = fin,
                                            annot = True,
                                            sous_mean = True,
                                            feature = "audio",
                                            f_max = 22000,
                                            overlap = 2, 
                                            energie = "both",
                                            precoce = True,
                                            power=2.0,
                                            nb_mels=64,
                                            derive1=True,
                                            derive2=False,
                                            nbr_classes= 2  )
                mels.append(xi)
                labels.append(yi)
            
            for debut,fin in zip([int(1*60)],[int(17*60 +40)]):   #zip([ind_mf4_train],[int(1*60)],[int(17*60+40)])
                xi,yi = fct.mels_features(list_mf4 = [ind_mf4_train],
                                            list_mics = ind_acc,
                                            N_window = N_window,
                                            debut = debut,
                                            fin = fin,
                                            annot = True,
                                            sous_mean = True,
                                            feature = "acc",
                                            f_max = 10000,
                                            overlap = 2, 
                                            energie = "both",
                                            precoce = True,
                                            power=2.0,
                                            nb_mels=64,
                                            derive1=True,
                                            derive2=False,                                                                                       
                                            nbr_classes= 2 )
                mels.append(xi)
                labels.append(yi)


            X,Y = [],[]
            X_val,Y_val = [],[]
            X_test,Y_test = [],[]

            for key in mels[0].keys():
                X,Y = np.array(mels[0][key]) , np.array(labels[0][key])
            for key in mels[1].keys():
                X,Y = np.concatenate((X,mels[1][key]),axis=1),Y
            

            print(f"shape des données est {X.shape} et la distribution des classes esr {np.bincount(Y)}")


        if not total : 
            mels.append(xi)
            labels.append(yi)
            X,Y = [],[]
            X_val,Y_val = [],[]
            X_test,Y_test = [],[]

            for key in mels[0].keys():
                X,Y = np.array(mels[0][key]),np.array(labels[0][key])

    
    #prétraitement des donnees
    X,Y,class_weights,pca,scaler = fct.pretrait(X,Y,n_components=0.95,
                                                apply_norm=True,
                                                seq=True,
                                                seq2seq=seq2seq,
                                                seq_apres=10,
                                                seq_before=9,
                                                apply_pca=apply_pca,
                                                overlap=0.5,
                                                super_vecteur=False
                                                )

    #Y[:, -1, 0]
    X_train,X_val,Y_train,Y_val = train_test_split(X,Y,stratify = Y,test_size=0.4,random_state=0)
    X_val,X_test,Y_val,Y_test = train_test_split(X_val,Y_val,stratify=Y_val,random_state=0,test_size=0.5)


    Y_train, Y_val, Y_test = to_categorical(Y_train,num_classes=nbr_classes),to_categorical(Y_val,num_classes=nbr_classes), to_categorical(Y_test,num_classes=nbr_classes)

    #importer le modele rcnn 
    input_shape = (X_train.shape[1],X_train.shape[2])

    nboutput = nbr_classes
    model = fct.CRNN_build(input_shape = input_shape,
                        NbOutNeur = nboutput,
                        NbFilterByLayer = NbrFilterByLayer,
                        NbNeurByRecuurLayer = NbrNeurByReccurLayer,
                        Bilstm = True,
                        seq2seq = seq2seq,
                        Use_dropout= True,
                        drop_out_rate= 0.1,
                        dense = True,
                        nbNeurByDenseLayer= neurByDense
                        )

    loss_focal = CategoricalFocalCrossentropy(alpha = class_weights,
                                             gamma=2.0,
                                             from_logits=False,
                                             label_smoothing=0.0,
        
        
                                             name='categorical_focal_crossentropy'
                                            )  

    lr = ExponentialDecay(
        initial_learning_rate = initial_learning_rate,
        decay_steps = 500,
        decay_rate = 0.96,
        staircase=True,
        name="ExponentialDecay",
    )
    optimizer = Adam(learning_rate = initial_learning_rate)

    model.summary()
    model.compile(optimizer = optimizer,
                loss = loss_focal,
                metrics = [AUCSeq(name="auc",num_classes=2),
                            F1ScoreSeq(name="f1_score",mode="frame",num_classes=nbr_classes),
                            #PrecisionSeq(name="Precision",mode="frame",num_classes=nbr_classes),
                            #RecallSeq(name="recall",mode="frame",num_classes=nbr_classes)
                        ])  


    earlystopping = EarlyStopping(monitor="val_f1_class_1",
                                mode = "max",
                                restore_best_weights= True,
                                patience = 10e+9,
                                start_from_epoch= 0)

    reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss',     # métrique à surveiller
                    factor=0.96,             # facteur de réduction du learning rate
                    patience=10,             # nombre d’époques sans amélioration avant réduction
                    min_lr=1e-7,            # learning rate minimum
                    verbose=0   ,            # affiche les messages lors des changements
                    mode = "min",
                    min_delta = 0.00000001
)


    history = model.fit(X_train,Y_train,
            batch_size=batch_size,
            verbose=1,
            callbacks = [earlystopping,reduce_lr],
            validation_data= (X_val,Y_val),
            epochs = epochs,
            shuffle = True)

    dossier = 'acceleros' if feature == "acc" else "audio"

    # Créer le dossier s'il n'existe pas
    save_dir = os.path.join('models', dossier)
    os.makedirs(save_dir, exist_ok=True)

    # Sauvegarde du modèle Keras
    chemin_save_keras = os.path.join(save_dir, f'{feature}_{"".join(map(str, mic))}.keras')
    model.save(chemin_save_keras)

    # Prédictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Sauvegarde des métriques
    metrique_keys = ["loss", "f1_class_1", "auc_binary",
                    "val_loss", "val_f1_class_1", "val_auc_binary"]

    history_metrique = {key: history.history[key] for key in metrique_keys}

    # Sauvegarde en pickle
    chemin_save_pkl = os.path.join(save_dir, f'{feature}_{"".join(map(str, mic))}.pkl')
    with open(chemin_save_pkl, "wb") as f:
        pickle.dump([[Y_train, y_pred_train],
                    [Y_val, y_pred_val],
                    [Y_test, y_pred_test],
                    [pca, scaler, history_metrique]], f)



print("done")