import os 
os.chdir("C:\\Users\\mohammed-amine.tanna\\OneDrive - RAILENIUM\\Bureau\\stage_railenium\\Telli\\Revigny")

from importlib import reload
import functions as fct
reload(fct)

import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import scipy.ndimage
debut, fin = int(1*60 + 10), int(17*60 + 40)
'''# Charger l'audio et les labels

dict_audio,_ ,sr = fct.read_waves([3], [1], debut=debut, fin=fin, sous_mean=True, feature="acc")
signal_audio = dict_audio["3_1"]
'''


'''
dict_labels, _ = fct.annotation([3], [1], nbr_classes=2, debut=debut, fin=fin, feature="acc")
labels_raw = np.array(dict_labels["3_1"])'''

# Paramètres
window_size = 2048   #2048
hop_size =  window_size//4    #512
'''
mels  = fct.fft_features(list_mics=[1],list_mf4=[4],N_window=window_size,debut=debut,fin=fin,nbr_classes=2,
                          sous_mean=True,feature="acc",f_max=20000,overlap=2,energie="both",annot=True) #,power=2.0,nb_mels=32,derive1=False,derive2=False




X1 = np.array(mels[0]["4_1"])
print(np.shape(np.array(X1)))
y1 = np.array(mels[1]["4_1"])
print(np.shape(np.array(y1)))


# Calcul des statistiques sur chaque échantillon (ligne)
max_vals = np.max(X1, axis=1)
median_vals = np.median(X1, axis=1)
mean_vals = np.mean(X1, axis=1)
min_vals = np.min(X1, axis=1)
sum_vals = np.sum(X1, axis=1)
std_vals = np.std(X1, axis=1)

# Fonction pour calculer la corrélation de Pearson
def corr(x, y):
    return np.corrcoef(x, y)[0, 1]

# Calcul des corrélations
corr_max = corr(max_vals, y1)
corr_median = corr(median_vals, y1)
corr_mean = corr(mean_vals, y1)
corr_min = corr(min_vals, y1)
corr_sum = corr(sum_vals, y1)
corr_std = corr(std_vals, y1)

# Affichage
stat_names = ['Max', 'Median', 'Mean', 'Min', 'Sum', 'Std Dev']
correlations = [corr_max, corr_median, corr_mean, corr_min, corr_sum, corr_std]

plt.figure(figsize=(10,5))
bars = plt.bar(stat_names, correlations, color='skyblue')
plt.title('Corrélation entre statistiques FFT et labelisation')
plt.ylabel('Coefficient de corrélation (Pearson)')
plt.ylim([0, 0.5])
plt.grid(axis='y')

# Afficher la valeur de corrélation au-dessus de chaque barre
for bar, corr_val in zip(bars, correlations):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.05, f'{corr_val:.2f}', ha='center', fontsize=10)

plt.show()'''

mels  = fct.mels_features(list_mics=[4],list_mf4=[4],N_window=window_size,debut=debut,fin=fin,nbr_classes=2,
                          sous_mean=True,feature="audio",f_max=22000,overlap=2,energie="both",annot=True,power=2.0,nb_mels=64)


X1 = np.array(mels[0]["4_4"])
print(np.shape(np.array(X1)))
y1 = np.array(mels[1]["4_4"])
print(np.shape(np.array(y1)))

X1,y1,_,_,_ = fct.pretrait(X1,y1,apply_norm=True,apply_pca=False,seq=False,super_vecteur=False)

energy_curve = []
aligned_labels = []

X = np.sum(X1[:,:-2],axis=1)
X,y = X , y1
energy_derivative = X
aligned_labels = y





import matplotlib.pyplot as plt
import librosa.display

fig, axes = plt.subplots(2, 1, figsize=(18, 10),sharex=True)
#librosa.display.specshow(mels[0]["4_3"][:,:16].T, ax=axes[0])
axes[0].plot(X)
axes[0].set_title("Spectrogramme Mel")
#axes[1].plot(range(len(aligned_labels)), energy_derivative,  color="green")
axes[1].step(range(len(aligned_labels)), aligned_labels, where='mid', color="orangered")
axes[1].set_title("Énergie et labels")


plt.ylabel("Valeurs")
plt.xlabel("Temps (s)")
plt.title("Dérivée de l'énergie et labels alignés")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
