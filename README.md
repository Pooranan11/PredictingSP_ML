# PredictingSP_ML

## 🇫🇷 Version française

### 1. Présentation  
Ce projet a pour objectif de **prédire l’évolution du cours d’un titre boursier** (dans l’exemple : NVIDIA – NVDA) en utilisant un modèle de Deep Learning de type **LSTM (Long Short-Term Memory)**.  
Le script télécharge les données historiques via `yfinance`, les normalise, crée des séquences temporelles, entraîne un réseau de neurones récurrent, et produit des prédictions du prix futur.

### 2. Fonctionnalités  
- Téléchargement automatique des données boursières via **Yahoo Finance** (`yfinance`).  
- Normalisation des prix avec **MinMaxScaler** (de `scikit-learn`).  
- Création de séquences glissantes sur 60 jours pour prédire le jour suivant.  
- Construction d’un modèle **LSTM** à l’aide de `tensorflow.keras.models.Sequential`.  
- Entraînement et affichage de la **perte (loss)** à chaque époque.  
- Vérification de l’environnement TensorFlow (GPU / CPU) via `test_tf_gpu.py`.  
- Scripts de test et d’expérimentation inclus.

### 3. Structure du dépôt  

PredictingSP_ML/
├── .gitignore
├── README.md ← ce fichier
├── requirements.txt ← dépendances Python
├── predict_nvda_lstm.py ← script principal de prédiction NVDA
├── predict_test.py ← script d’expérimentation
├── test.py ← script de test simple
├── test_tf_gpu.py ← test de compatibilité TensorFlow
└── ...

### 4. Installation & exécution  
1. **Cloner le dépôt :**  
   ```bash
   git clone https://github.com/Pooranan11/PredictingSP_ML.git
   cd PredictingSP_ML

2. **Créer et activer un environnement virtuel (Python 3.11 recommandé) :**
    python3.11 -m venv .venv-tf
    source .venv-tf/bin/activate
