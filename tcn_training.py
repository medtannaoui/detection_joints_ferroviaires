import random
random.seed(0)

import os
from importlib import reload
import functions as fct
reload(fct)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryFocalCrossentropy,CategoricalFocalCrossentropy
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import train_test_split
from metric_seq2seq import PrecisionSeq, RecallSeq, AUCSeq, F1ScoreSeq
import numpy as np
import pickle

# ===== Paramètres =====
ind_mf4_train = [4]   # indices des fichiers mf4
ind_acc = [1]         # indices des capteurs accéléro
sr = 100000           # fréquence d'échantillonnage
window_size = fct.get_near_pof2(0.02,sr)    # taille des fenêtres en échantillons
overlap = 0.5         # recouvrement
nbr_classes = 2
batch_size = 128
epochs = 20
initial_learning_rate = 1e-4

# ===== Fonction découpe signal brut en fenêtres =====
def segment_signal(signal, labels, window_size, overlap):
    step = int(window_size * (1 - overlap))
    X, Y = [], []
    for i in range(0, len(signal) - window_size, step):
        X.append(signal[i:i+window_size])
        # on prend le label majoritaire dans la fenêtre
        Y.append(int(np.round(np.mean(labels[i:i+window_size]))))
    X = np.array(X)[..., np.newaxis]  # shape: (nb_windows, window_size, 1)
    Y = np.array(Y)
    return X, Y

# ===== Lecture du signal brut =====
# read_waves doit retourner signal (np.array) et labels (np.array)
raw_signal= fct.read_waves(list_mf4=ind_mf4_train,
                                        list_mics=ind_acc,
                                        debut=int(1*60),
                                        fin=int(17*60+40),
                                        feature="acc")[0][f"{ind_mf4_train[0]}_{ind_acc[0]}"]
raw_labels = fct.annotation(list_mf4=ind_mf4_train,list_mics=ind_acc,nbr_classes=nbr_classes,debut=60,fin=int(17*60+40),feature="acc")[0][f"{ind_mf4_train[0]}_{ind_acc[0]}"]

# Segmentation
X, Y = segment_signal(raw_signal, raw_labels, window_size, overlap)
print(f"Shape X: {X.shape}, Distribution classes: {np.bincount(Y)}")

# Normalisation
mean = np.mean(X, axis=0)
std = np.std(X, axis=0) + 1e-8
X = (X - mean) / std

# Split train/val/test
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, stratify=Y, test_size=0.4, random_state=0)
X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, stratify=Y_val, test_size=0.5, random_state=0)

Y_train = to_categorical(Y_train, num_classes=nbr_classes)
Y_val   = to_categorical(Y_val, num_classes=nbr_classes)
Y_test  = to_categorical(Y_test, num_classes=nbr_classes)

# ===== Modèle TCN =====


model = fct.TCN_build(input_shape=(window_size, 1), nb_classes=nbr_classes,dilations=[2])

# ===== Entraînement =====
loss_focal = CategoricalFocalCrossentropy(alpha = 1.0,
                                             gamma=2.0,
                                             from_logits=False,
                                             label_smoothing=0.0,
        
        
                                             name='categorical_focal_crossentropy'
                                            )  
optimizer = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer=optimizer, loss=loss_focal,
              metrics=[AUCSeq(name="auc",num_classes=2),
                       F1ScoreSeq(name="f1_score",mode="frame",num_classes=nbr_classes)])

model.summary()

earlystopping = EarlyStopping(monitor="val_f1_class_1",
                               mode="max",
                               restore_best_weights=True,
                               patience=10e+9)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=5,
                              min_lr=1e-6,
                              verbose=1)

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    validation_data=(X_val, Y_val),
                    epochs=epochs,
                    callbacks=[earlystopping, reduce_lr])

# ===== Sauvegarde =====
chemin_save_keras = os.path.join('models', 'feature', f'acc_tcn.keras')
model.save(chemin_save_keras)

# Prédictions
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

chemin_save_pkl = os.path.join('models', 'feature', f'acc_tcn.pkl')
with open(chemin_save_pkl, "wb") as f:
    pickle.dump([[Y_train, y_pred_train],
                 [Y_val, y_pred_val],
                 [Y_test, y_pred_test],
                 [mean, std, history.history]], f)

print("done")
