Détection des joints à partir des audios et des Accéléromètres.
-----------------------

# Introduction

Ce projet vise à développer un système capable de détecter automatiquement les joints présents sur une voie
en analysant des enregistrements audio. Les fichiers audio, associés à des annotations labellisées via Audacity, 
contiennent des informations acoustiques permettant d’identifier ces irrégularités. L’objectif est de traiter 
ces données pour extraire les caractéristiques sonores typiques des joints, afin d’automatiser leur repérage et 
ainsi améliorer les processus d’inspection et de maintenance.

# Installation

```bash
pip install librosa==0.10.1
pip install scipy==1.11.4
pip install numpy==1.26.2
pip install tensorflow==2.15.0
pip install scikit-learn==1.3.2
pip install matplotlib==3.8.2
pip install pandas==2.1.3
pip install seaborn==0.13.0
pip install ipython==8.18.1
```

# Structure du projet
```bash
/
│
├── data/
│   ├── acceleros/              # les fichiers .wav extraits des canaux des accéléromètres.
│   ├── audios/                 # les fichiers .wav issus des enregistrements audio principau.
│   ├── images/                 # les images linéaires extraites ou générées à partir des signaux.
│   ├── images_fusionée/        # trois grandes images liés à chaque mf4 
│   └── annotation_audacity/    # les fichiers d’annotations (.txt) générés avec Audacity pour chaque .mf4.
│
├── models/                     # Des ficher pkl qui contient les paramétres d'entrainement et des fichiers keras qui contient les poids des modèles.
│   ├──acceleros/                   
│   ├──audios/                    
│   ├──fusion/  
│
├── resultats/                  # les résultats après entrainement (matrice de confusion , courbe roc , f1 score sur les trois sets ...).
│   ├──audios/                    
│   ├──acceleros/
│   ├──fusion/  
│
├── visualistaion_history/      # Des visualisation des caractéristiques (comme l'énergie et sa dérives en fonctions des labels).
│
├── vitesses/                   # Des fichiers pkl qui représente pour chaque mf4 la vitesse en fonction de temps et des images des vitesses.
│
├── functions.py/               # Tous les fonctions d'extraction des caractéristques et de prétraitement et des architectures neuronnaux et les fonction de validation.
│
├── methodes_fusion.py/         # Contient l'implémentation des méthodes de fusion tardive.
│
├── metric_seq2seq.py/          # Contient les metriques codés d'une façon tel qu'il marche pour la prédiction séquence séquence et trame trame.
│
├── train_joint.py/             # fichier ou l'entrainement est lancé.
│
├── test_fusion.py/             # le fichier ou j'applique de la fusion tardive.
│
├── visualisation.py/           # fichier ou je fais la visualisation des caractéristques avant l'entrainmenet.
│
├── tcn_training.py/           # fichier ou l'entrainemnet avec le tcn est lancé
│
├── ml_classique.ipynb/         # notebook ou j'essaye au debut avec des modèles de machine learning classiques.
│
└── README.md                   # Description du projet
```

# Utilisation

## Entrainement
```bash
Pour lancer un entrainement il suffit de modifier le fichier train_joint :
    - Fixer les paramétres des caractéristiques ( les variables situés en haut du code ).
    - Fixer les paramétres du modèles à utiliser ( nombre de filtre, nombre de couches, taux d'apprentissage utilisé, batch size , etc... )
    - Faire un entrainmenet avec un TCN sur le signal brut en controlan la liste des dilations avec le fichier tcn_train.py.
```
## Après entrainement

```
- Faire des prédictions sur les données de test avec le notebook test_prediction.ipynb.
- Essayer de la fusion tardive avec le ficher test_fusion.py on selectionnant les indices du modèles à fusionner.
```

 