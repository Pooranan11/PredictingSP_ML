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
    ```bash
    python3.11 -m venv .venv-tf
    source .venv-tf/bin/activate

3. **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt

4. **Exécuter le script principal :**
    ```bash
    python predict_nvda_lstm.py

(Vous pouvez aussi lancer python test_tf_gpu.py pour vérifier la bonne installation de TensorFlow.)

### 5. Paramètres importants

Ticker : modifiable dans predict_nvda_lstm.py → ticker = 'NVDA'.
Période d’analyse : modifiable via les arguments start et end dans yf.download().
Longueur des séquences : sequence_length = 60 (modifiable).

### 6. Résultats attendus

Affichage de la progression d’entraînement (Epoch X/84 - loss: ...).
Prédiction d’un prix futur basé sur les 60 derniers jours.
Possibilité d’étendre à d’autres titres (AAPL, MSFT, TSLA, etc.).

### 7. Limitations et avertissements

⚠️ Ce projet est destiné uniquement à des fins éducatives.
Les résultats ne doivent en aucun cas être considérés comme des conseils d’investissement.
Les performances du modèle dépendent de nombreux facteurs : la période, les données, les hyperparamètres, etc.

### 8. Contribution

Les contributions sont les bienvenues !
Vous pouvez :
ajouter de nouveaux tickers ;
améliorer l’architecture du modèle (plusieurs couches LSTM, Dropout, etc.) ;
ajouter des métriques ou des visualisations.
Proposez vos idées via une Issue ou une Pull Request.