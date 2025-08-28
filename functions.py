'''
fichier contenant tous les fonctions d'extractions des features et de pretraitement et de visualisation et d'entrainement et de validation
'''
import os 
#importation des bibs de traitement de signal
import librosa 
from librosa.feature import melspectrogram
from scipy.fft import fft , fftfreq
from scipy.signal import periodogram
from scipy.signal import hilbert
import numpy as np 
from IPython.display import Audio
import scipy 
from numpy.random import uniform as rndm
from scipy.io.wavfile import write
import random

from tensorflow.keras import layers, Model, Input
#importation des bibs de tensorflow et scikit-learn
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,LSTM,Dropout,Bidirectional,Conv1D,Conv2D,GRU,LeakyReLU,Input,Reshape,BatchNormalization,add,GlobalAveragePooling1D,ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint 
from tensorflow.keras.losses import BinaryFocalCrossentropy
from sklearn.metrics import confusion_matrix,precision_score,recall_score,precision_recall_curve,classification_report,make_scorer, f1_score,roc_auc_score
from tensorflow.keras.metrics import Precision,Recall,F1Score,AUC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight,compute_sample_weight
from sklearn.preprocessing import StandardScaler , MinMaxScaler

#bibs pour la visaulisation
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
import sys

#extraction des features
def get_near_pof2(val, sr):
    """
    Calcule la puissance de 2 la plus proche de la durée (val) convertie en nombre d'échantillons.

    Paramètres :
    - val (float) : Durée de la fenêtre souhaitée (en secondes).
    - sr (int) : Fréquence d'échantillonnage (en Hz).

    Retour :
    - int : Nombre d'échantillons correspondant à la puissance de 2 la plus proche.
    """
    value = int(sr*val)

    p2 = np.log2(value)
    roundp2 = np.round(p2)

    return int(2**roundp2)

def read_waves(list_mf4: list, list_mics: list,debut=0,fin=-1,plot=False,sous_mean=True,feature = "audio") -> tuple:
    """
    Lit les fichiers audio et retourne les signaux correspondants.

    Paramètres :
    - list_mf4 (list) : Liste des indices des fichiers mf4 à utiliser.
    - list_mics (list) : Liste des indices des microphones à utiliser.

    Retour :
    - tuple :
        - dict : Dictionnaire des signaux audio (clé = "mf4_mic").
        - dict : Dictionnaire des tailles de signaux pour chaque canal.
        - int : Fréquence d échantillonnage
    """

    signals = {}
    N = {}

    if feature == "audio" : 
        for i in list_mf4:
            for j in list_mics:
                
                path = os.path.join('data','audios',f'Son_{i}_{j}.wav')  #f"{os.getcwd()}\\data\\audios\\Son_{i}_{j}.wav"
                
                signals[f"{i}_{j}"], sr = librosa.load(path, sr=None)

                debut_sample = int(sr * debut)
                if fin == -1:
                    fin_sample = None
                else:
                    fin_sample = int(sr * fin)

                signals[f"{i}_{j}"] = signals[f"{i}_{j}"][debut_sample:fin_sample]

                if sous_mean:
                    signals[f"{i}_{j}"] -= np.mean(signals[f"{i}_{j}"])

                N[f"{i}_{j}"] = len(signals[f"{i}_{j}"])
    if feature == "acc" :    
        for i in list_mf4:
            for j in list_mics:
                
                path =   os.path.join('data','acceleros',f'Acc_{i}_{j}.wav') #f'{os.getcwd()}\\data\\acceleros\\Acc_{i}_{j}.wav'
                
                signals[f"{i}_{j}"], sr = librosa.load(path, sr=None)

                debut_sample = int(sr * debut)
                if fin == -1:
                    fin_sample = None
                else:
                    fin_sample = int(sr * fin)

                signals[f"{i}_{j}"] = signals[f"{i}_{j}"][debut_sample:fin_sample]

                if sous_mean:
                    signals[f"{i}_{j}"] -= np.mean(signals[f"{i}_{j}"])

                N[f"{i}_{j}"] = len(signals[f"{i}_{j}"])
            
    if plot : 
        
        for i in list_mf4 : 
            for j in list_mics : 
                plt.figure(figsize=(13,3))
                librosa.display.waveshow(signals[f'{i}_{j}'],sr=sr)
        
    return signals, N, sr

def annotation(list_mf4: list, list_mics: list, nbr_classes: int,debut=0,fin=-1,label="joint",feature="audio") -> tuple:
    """
    Lit les annotations et crée un dictionnaire de labels pour chaque signal pour chaque echantillon.

    Paramètres :
    - list_mf4 (list) : Liste des fichiers mf4 à traiter.
    - list_mics (list) : Liste des microphones.
    - nbr_classes (int) : Nombre de classes (2 ou 3).
    - sr (int) : Fréquence d'échantillonnage.

    Retour :
    - tuple :
        - dict : Dictionnaire des annotations pour chaque canal.
        - list : Liste de DataFrames des fichiers d'annotations originaux.
    """
    annotations = {}
    datas = []
    signals,_,sr=read_waves(list_mf4=list_mf4,debut=debut,fin=fin,list_mics=list_mics,feature=feature)
    for i in list_mf4:
        if nbr_classes==3 and label != "joint":
            path =  os.path.join('data','annotations_audacity',f'data_ann{i}.txt')        #f"{os.getcwd()}\\data\\annotations_audacity\\data_ann{i}.txt"
        else :
            path = os.path.join('data','annotations_audacity',f'annotations_{i}.txt')         #f"{os.getcwd()}\\data\\annotations_audacity\\annotations_{i}.txt"

        data = pd.read_csv(path, sep='\t', header=None)
        data.columns = ["debut", "fin", "label"]
        datas.append(data)

        for j in list_mics:
            
            annotations[f"{i}_{j}"] = np.zeros(len(signals[f"{i}_{j}"]))

        for _, row in data.iterrows():
            debut_indx = int((row["debut"]) * sr) - (debut * sr)
            fin_indx = int((row["fin"]) * sr) - ( debut*sr)
            label = row["label"]

            for j in list_mics:
                key = f"{i}_{j}"
                if nbr_classes == 2 : #and label in ["avg","avd","avg+avd","avd+avg"]          #pour s entrainer seulement sur les roues en avant on ajoute cette condition
                    annotations[key][debut_indx:fin_indx] = 1
                elif nbr_classes == 3:
                    if label in ["avg","avd","avg+avd","avd+avg"]:
                        annotations[key][debut_indx:fin_indx] = 1
                    elif label in ["ard","arg","arg+ard","ard+arg"]:
                        annotations[key][debut_indx:fin_indx] = 2
    return annotations, datas

#calcule des coefs mels

def compute_energy_curve(signal, window_size=2048, hop_size=512, mode="energy"):
    energy_curve = []
    for start in range(0, len(signal) - window_size + 1, hop_size):
        end = start + window_size
        frame = signal[start:end]
        energy = np.sum(frame ** 2) / window_size
        energy_curve.append(energy)
    
    energy_curve = np.array(energy_curve)
    
    if mode == "derivative":
        energy_curve = np.diff(energy_curve, prepend=energy_curve[0])
    
    return energy_curve.reshape(-1, 1)  # Assure une shape (n_frames, 1)

def spectrogram(y, NFFT=2048, HOP_LENGTH=512, WIN='hann', FS=100000, Flag_DSP=True):
    DSP = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP_LENGTH, window=WIN, win_length=NFFT, center=True)) ** 2
    #DSP /= np.sum(DSP,axis=1)
    Frame2time = librosa.frames_to_time(np.arange(DSP.shape[1]), sr=FS, hop_length=HOP_LENGTH)
    PW = np.sum(DSP, axis=0)
    if Flag_DSP:
        return DSP, PW, Frame2time
    else:
        return np.abs(DSP), PW, Frame2time

def get_MELgrame(audio, Nt, NB_FEATURES=64, FMAX=20000, power=2.0, FS=100000,
                 overlap=4, energie=True, derive1=False, derive2=False):
    
    hop_length = Nt // overlap

    # Spectrogramme de puissance
    S, _, tprocess = spectrogram(audio, NFFT=Nt, HOP_LENGTH=hop_length, WIN='hann', FS=FS, Flag_DSP=True)

    # Melspectrogramme
    melspectrograms = librosa.feature.melspectrogram(sr=FS, S=S, n_mels=NB_FEATURES, fmax=FMAX, fmin=1)
    Logmels = 20.0 / power * np.log10(melspectrograms + sys.float_info.epsilon)

    n_frames = Logmels.shape[1] - overlap  # Pour enlever les dernières frames

    feature_list = []
    feature_list.append(Logmels[:, :n_frames].T)  # (n_frames, NB_FEATURES)

    # Dérivées de mels
    if derive1:
        delta1 = librosa.feature.delta(Logmels)
        feature_list.append(delta1[:, :n_frames].T)
    if derive2:
        delta2 = librosa.feature.delta(Logmels, order=2)
        feature_list.append(delta2[:, :n_frames].T)

    # Energie ou dérivée d'énergie
    if energie is True or (isinstance(energie, str) and energie.lower() in ["energy", "both", "derivative"]):
        energy_curve = compute_energy_curve(audio, window_size=Nt, hop_size=hop_length, mode="energy")
        feature_list.append(energy_curve[:n_frames])  

    if isinstance(energie, str) and energie.lower() in ["derivative", "both"]:
        energy_deriv = compute_energy_curve(audio, window_size=Nt, hop_size=hop_length, mode="derivative")
        feature_list.append(energy_deriv[:n_frames])  

    # Empilement final
    full_features = np.hstack(feature_list)  # Shape: (n_frames, total_features)

    return full_features.T, tprocess[:n_frames]


def fft_features(list_mics,list_mf4,N_window,debut=0,
                  fin=None,nbr_classes=None,annot=False,sous_mean=True,feature="audio",
                  f_max=22000,overlap=2,
                  energie=True,precoce=False,universel=False,
                  label="joint"):
    ''' 
    Extrait les coefs de mel des signaux audio.

    Paramètres :
    - list_mf4 (list) : Liste des indices des fichiers mf4.
    - list_mics (list) : Liste des indices des microphones.
    - N_window (int) : Taille de la fenêtre principale.
    - nbr_classes (int) : Nombre de classes (2 ou 3).
    - annot : boolean pour specifier est ce qu on retourne aussi le dictionnaire d'annotations

    Retour :
    - tuple :
        - dict : dictionnaire contient les coefs de mels pour chaque canal.
        - dict : Labels associés à chaque fenêtre.    
    '''
    ffts = {}
    labels = {}
    keys = []
    signals , N , sr = read_waves(list_mf4=list_mf4,list_mics=list_mics,debut=debut,fin=fin,sous_mean=sous_mean,feature=feature)
    if annot == True:
        if label == "joint" : 
            annotations ,_ = annotation(list_mf4=list_mf4,list_mics=list_mics,nbr_classes=nbr_classes,debut=debut,fin=fin,label="joint",feature=feature)
        else : 
            annotations ,_ = annotation(list_mf4=list_mf4,list_mics=list_mics,nbr_classes=nbr_classes,debut=debut,fin=fin,label="2",feature=feature)


    for i in list_mf4 : 
        for j in list_mics : 
            keys.append(f'{i}_{j}')
            
    

 

    for key in keys : 
        audio = signals[key]
        ffts[key] = []
        labels[key] = []
        

        for db in range(0, len(audio) - N_window + 1  ,N_window//overlap):
            segment = audio[db:db+(N_window//overlap)]
            label = annotations[key][db:db+(N_window//overlap)]

            segment -= np.mean(segment)
            if not energie : 
                segment /= np.std(segment) + 1e-15
            
            fft_segment = np.fft.fft(segment,n=N_window)
            fft_segment = np.abs(fft_segment)
            segment = np.log10(fft_segment + 1e-15)

            lb = 1 if 1 in label else 2 if 2 in label else 0
            ffts[key].append(fft_segment)
            labels[key].append(lb)

    
        if annot : 
            min_len = min(len(ffts[key]), len(labels[key]))
            ffts[key] = ffts[key][:min_len]
            labels[key] = labels[key][:min_len]

    
    if precoce or universel :                           #pour les fusion horizontale et verticale
        axis = 1 if precoce else 0
        col_name = "precoce" if precoce else "universel"
        X,y={},{}
        for m,key in enumerate(ffts.keys()):
            if m==0:
                X[col_name]=ffts[key]
                y[col_name]=labels[key]
            else :
                X[col_name] = np.concatenate((X[col_name],ffts[key]),axis=axis)
                if universel : 
                    y[col_name] = np.concatenate((y[col_name],labels[key]),axis=axis)
                 

        ffts = X
        labels = y

        
    return ffts , labels       #retourn le dictionnaire des mels et le dictionnaire des labelisations


def mels_features(list_mics,list_mf4,N_window,power,nb_mels,debut=0,
                  fin=None,nbr_classes=None,annot=False,sous_mean=True,feature="audio",
                  f_max=22000,overlap=2,
                  energie=True,precoce=False,universel=False,
                  coefs="mels",n_mfcc=15,derive1=False,derive2=False,label="joint"):
    ''' 
    Extrait les coefs de mel des signaux audio.

    Paramètres :
    - list_mf4 (list) : Liste des indices des fichiers mf4.
    - list_mics (list) : Liste des indices des microphones.
    - N_window (int) : Taille de la fenêtre principale.
    - nbr_classes (int) : Nombre de classes (2 ou 3).
    - annot : boolean pour specifier est ce qu on retourne aussi le dictionnaire d'annotations

    Retour :
    - tuple :
        - dict : dictionnaire contient les coefs de mels pour chaque canal.
        - dict : Labels associés à chaque fenêtre.    
    '''
    mels = {}
    labels = {}
    keys = []
    signals , N , sr = read_waves(list_mf4=list_mf4,list_mics=list_mics,debut=debut,fin=fin,sous_mean=sous_mean,feature=feature)
    if annot == True:
        if label == "joint" : 
            annotations ,_ = annotation(list_mf4=list_mf4,list_mics=list_mics,nbr_classes=nbr_classes,debut=debut,fin=fin,label="joint",feature=feature)
        else : 
            annotations ,_ = annotation(list_mf4=list_mf4,list_mics=list_mics,nbr_classes=nbr_classes,debut=debut,fin=fin,label="2",feature=feature)


    for i in list_mf4 : 
        for j in list_mics : 
            keys.append(f'{i}_{j}')
            
    #parcourir tous les canals
    methode = None

 

    for key in keys : 
        methode =  get_MELgrame(audio = signals[key],Nt=N_window,NB_FEATURES=nb_mels,FMAX=f_max,power=2.0,FS=sr,overlap=overlap,energie=energie,derive1=derive1,derive2=derive2)[0].T #if coefs =="mels" else get_MFCCgrame(signals[key], N_window, HL=50, NB_MFCC=n_mfcc, FMAX=22000, power=2.0, FS=sr,overlap=overlap)[0].T

        mels[key] = methode
       
        #labelisation
        if annot is not False : 
            labels[key] = []
            hop = N_window // overlap
            for i in range(0, N[key] - N_window+1 , hop):
                i += debut
                window_label = annotations[key][i:i+N_window]
                if len(window_label) < N_window : 
                    
                    break           #sauter le dernier echantillon

                if nbr_classes== 2 :
                    if 1 in window_label : 
                        label = 1
                    else : 
                        label = 0
                elif nbr_classes == 3 :
                    if 1 in window_label : 
                        label = 1
                    elif 2 in window_label : 
                        label = 2
                    else : 
                        label =0
                labels[key].append(label)
                #f=i+N_window
        if annot : 
            min_len = min(len(mels[key]), len(labels[key]))
            mels[key] = mels[key][:min_len]
            labels[key] = labels[key][:min_len]

    
    if precoce or universel :                           #pour les fusion horizontale et verticale
        axis = 1 if precoce else 0
        col_name = "precoce" if precoce else "universel"
        X,y={},{}
        for m,key in enumerate(mels.keys()):
            if m==0:
                X[col_name]=mels[key]
                y[col_name]=labels[key]
            else :
                X[col_name] = np.concatenate((X[col_name],mels[key]),axis=axis)
                if universel : 
                    y[col_name] = np.concatenate((y[col_name],labels[key]),axis=axis)
                 

        mels = X
        labels = y

        
    return mels , labels       #retourn le dictionnaire des mels et le dictionnaire des labelisations


#pretraitement des données
import numpy as np

def create_data_sequences(x_data, y_data=None, num_sequence_behind=0,
                          num_sequence_ahead=0, overlap=0.5, seq2seq=False):
    """
    Crée des séquences temporelles avec contexte à partir des données d'entrée.

    Paramètres :
    - x_data : np.array, forme (n_samples, n_features)
    - y_data : np.array optionnel, forme (n_samples,)
    - num_sequence_behind : nombre de trames de contexte avant
    - num_sequence_ahead : nombre de trames de contexte après
    - overlap : taux de chevauchement entre séquences (0 à <1)
    - seq2seq : si True, chaque trame a un label ; sinon un seul label par séquence

    Retour :
    - sequences : np.array, (n_sequences, total_len, n_features)
    - targets : np.array, (n_sequences,) ou (n_sequences, total_len, 1)
    """
    sequences = []
    targets = []

    x_data = np.array(x_data)
    if y_data is not None:
        y_data = np.array(y_data)

    total_len = num_sequence_behind + num_sequence_ahead + 1
    stride = max(int((1 - overlap) * total_len), 1)

    for center_idx in range(num_sequence_behind,
                            len(x_data) - num_sequence_ahead,
                            stride):

        start_idx = center_idx - num_sequence_behind
        end_idx = center_idx + num_sequence_ahead + 1

        seq = x_data[start_idx:end_idx]
        if seq.shape[0] != total_len:
            continue

        sequences.append(seq)

        if y_data is not None:
            label_seq = y_data[start_idx:end_idx]

            if seq2seq:
                targets.append(label_seq.reshape(-1, 1))
            else:
                label = 1 if np.any(label_seq == 1) else 0
                targets.append(label)

    sequences = np.array(sequences)
    targets = np.array(targets) if y_data is not None else None

    print(f"{len(sequences)} séquences créées avec un chevauchement de {(1 - overlap) * 100:.1f}%.")

    return (sequences, targets) if y_data is not None else sequences




def pretrait(X, Y, n_components=0.95, apply_pca=False, seq=True,
             pca=None, scaler=None, list_classes_weights=None, super_vecteur=False,
             seq_before=3, seq_apres=3,
             apply_norm=False,
             overlap=0.5, seq2seq=False):
    """
    Effectue diverses opérations de prétraitement sur les données de caractéristiques.

    Paramètres :
    - X (np.array) : Les données de caractéristiques (Mel-spectrogrammes, etc.).
    - Y (np.array) : Les labels associés aux données.
    - n_components (float ou int) : Nombre de composantes pour PCA (ex: 0.95 pour 95% variance ou int).
    - pca_before_seq (bool) : Appliquer PCA avant la création de séquences.
    - norm_before_seq (bool) : Appliquer normalisation min-max avant la création de séquences.
    - seq (bool) : Créer des séquences.
    - pca (sklearn.decomposition.PCA) : Objet PCA déjà entraîné pour transformation (pour données de test).
    - scaler (sklearn.preprocessing.MinMaxScaler) : Objet MinMaxScaler déjà entraîné.
    - list_classes_weights (np.array) : Poids de classe pré-calculés.
    - super_vecteur (bool) : Si True, les séquences sont aplaties en un "super vecteur".
    - seq_before (int) : Nombre de trames de contexte avant la fenêtre centrale (pour create_data_sequences).
    - norm_before_pca (bool) : Appliquer normalisation min-max avant PCA.
    - overlap (float) : Chevauchement des séquences.
    - seq2seq (bool) : Labelisation séquence à séquence.

    Retour :
    - tuple : (X_processed, Y_processed, class_weights, pca_model, scaler_model)
    """
    # 1. Super Vecteur (optionnel)
    if super_vecteur:
        X, Y = create_data_sequences(X, Y,
                                     num_sequence_behind=1,
                                     num_sequence_ahead=1,
                                     overlap=0.5,seq2seq=False)
        # Aplati en un "super-vecteur"
        X = X.reshape(X.shape[0], -1)
        print(f"Forme après création de super-vecteur : {X.shape}")


    # 2. Normalisation avant PCA (optionnel)
    if apply_norm:
        if scaler is None: # Entraîner le scaler si c'est la première fois
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X,Y)
            
            print("Normalisation par rapport au dataset est appliquée aux données d'entrainemnet")
        else: # Utiliser le scaler déjà entraîné (sur le set de validation et de test)
            
            X = scaler.transform(X)
            print("Normalisation par rapport au dataset est appliquée aux données de validation")

            


    # 3. PCA avant séquences (optionnel)
    if apply_pca:
        if pca is None:
            pca = PCA(n_components=n_components, whiten=True)
            X_pca = pca.fit_transform(X, Y)
            # Vérification et ajustement du nombre de composants si trop bas
            if X_pca.shape[1] < 10 and n_components != X_pca.shape[1]: # n_components peut être un int ou float
                 print(f"PCA avec n_components={n_components} a résulté en {X_pca.shape[1]} composants. Forçant à 10.")
                 pca = PCA(n_components=min(10, np.array(X).shape[1]), whiten=True) # Max 10 ou moins si moins de features
                 X = pca.fit_transform(X,Y)
            else:
                 X = X_pca
            
            print(f"PCA entraînée et appliquée aux données d'entrainement - Nouvelle forme: {X.shape}")
        else:
            X = pca.transform(X)
            print("PCA appliquée sur les données de validation ")

    # 5. Création de Séquences 
    if seq : 
        X, Y = create_data_sequences(X, Y,
                                     num_sequence_behind=seq_before,
                                     num_sequence_ahead=seq_apres,
                                     overlap=overlap, seq2seq=seq2seq)
        print(f"Forme après création de séquences : {X.shape}")

        if seq2seq:
            # reshape pour LSTM en output seq to seq (avec 1 feature de label par trame)
            Y = Y.reshape(Y.shape[0], Y.shape[1], 1)
        else:
            Y = Y.reshape(-1, 1) # Pour la classification de séquence

    # Calcul des poids de classe (si non fournis)
    if list_classes_weights is None:
        # Y doit être un tableau 1D pour compute_class_weight
        if seq2seq == False:
            Y_flat = Y.ravel() if np.array(Y).ndim > 1 else Y
            
        else : 
            Y_flat = np.argmax(Y, axis=-1).flatten()

        list_classes_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(Y_flat),
                y=Y_flat
            )
        print(f"Poids de classe calculés {list_classes_weights}")
    else:
        print(f"Poids de classe fournis déja {list_classes_weights}")

    return X, Y, list_classes_weights, pca, scaler



#modéles d'entrainemnet 


def CRNN_build(input_shape, 
                     NbOutNeur, 
                     NbFilterByLayer, 
                     NbNeurByRecuurLayer, 
                     Use_dropout=False, 
                     drop_out_rate=0.3, 
                     Bilstm=True,
                     seq2seq=False,
                     dense=True,
                     nbNeurByDenseLayer = [16]):
    inputs = Input(shape=input_shape)  # (n_steps, n_features)
    x = Reshape((input_shape[0], input_shape[1], 1))(inputs)  # -> (n_steps, n_features, 1)

    # --- CNN avec strides pour réduire fréquence, pas le temps ---
    for numfilt,nb_filters in enumerate(NbFilterByLayer):
        x = Conv2D(filters=nb_filters, kernel_size=(5 if numfilt==0 else 3, 3), strides=(1, 2 if numfilt == len(NbFilterByLayer)-1 else 1), padding='same',name="conv2D_"+str(numfilt))(x)
        x = LeakyReLU(alpha=0.1)(x)
        if Use_dropout:
            x = Dropout(drop_out_rate)(x)

    # --- Préparer pour LSTM ---
    shape = x.shape  # (batch, n_steps, freq_reduced, channels)
    x = Reshape((shape[1], shape[2] * shape[3]))(x)  # fusionne freq_reduced × channels

    # --- LSTM Layers ---
    for numreccur,units in enumerate(NbNeurByRecuurLayer[:-1]):
        if Bilstm:
            x = Bidirectional(LSTM(units, return_sequences=True,name="LSTM_"+str(numreccur)))(x)
        else:
            x = LSTM(units, return_sequences=True,name="LSTM_"+str(numreccur))(x)
        
        x= LeakyReLU(alpha=0.1)(x)

        if Use_dropout:
            x = Dropout(drop_out_rate)(x)
            
    #derniere couche
    if Bilstm:
        x = Bidirectional(LSTM(NbNeurByRecuurLayer[-1],return_sequences=seq2seq,name="LSTM_"+str(len(NbNeurByRecuurLayer)-1)))(x)
    else: 
        x = LSTM(NbNeurByRecuurLayer[-1],return_sequences=seq2seq,name="LSTM_"+str(len(NbNeurByRecuurLayer)-1))(x)

    x = LeakyReLU(alpha=0.1)(x)

    if Use_dropout:
        x = Dropout(drop_out_rate)(x)

    if dense : 
        for nbrneur in nbNeurByDenseLayer : 
            x = Dense(nbrneur,activation="relu")(x)

    # --- Output ---
    
    output = Dense(NbOutNeur, activation='softmax')(x)


    model = Model(inputs, output,name="Detection_Joints")
    
    
    
    return model


#TCN 


def TCN_build(input_shape, nb_classes, nb_filters=64, kernel_size=5, dilations=[2], dropout_rate=0.2):
    inputs = Input(shape=input_shape)
    x = inputs
    for d in dilations:
        residual = x
        x = layers.Conv1D(filters=nb_filters, kernel_size=kernel_size,
                          padding='causal', dilation_rate=d)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv1D(filters=nb_filters, kernel_size=kernel_size,
                          padding='causal', dilation_rate=d)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout_rate)(x)
        if residual.shape[-1] != nb_filters:
            residual = layers.Conv1D(nb_filters, kernel_size=1, padding='same')(residual)
        x = layers.add([residual, x])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(nb_classes, activation='softmax')(x)
    return Model(inputs, outputs, name="TCN_Accelero")



from PIL import Image

def concat_images(start_idx, end_idx,indice_mf4 = 4):
    """
    Concatène les images BMP d'un dossier (par ordre trié) en une seule image.

    """
    folder_path = os.path.join("data","images")
    output_path = os.path.join("data","images_fusionée",f"mf4_{indice_mf4}","image.bmp")
    # Récupérer la liste des images triées
    all_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".bmp")])
    
    # Sélectionner seulement l'intervalle demandé
    selected_files = all_files[start_idx:end_idx+1]
    if not selected_files:
        raise ValueError("Aucune image trouvée dans l'intervalle demandé.")
    
    # Charger toutes les images
    images = [Image.open(os.path.join(folder_path, f)) for f in selected_files]

    # Vérifier dimensions (on suppose toutes les images de même taille)
    widths, heights = zip(*(img.size for img in images))

    # Concaténation verticale (pile les images les unes en dessous des autres)
    total_height = sum(heights)
    max_width = max(widths)

    new_im = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]

    # Sauvegarde
    new_im.save(output_path)
    print(f"✅ Image concaténée sauvegardée : {output_path}")
