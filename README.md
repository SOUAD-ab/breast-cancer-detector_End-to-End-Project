
# Détection du Cancer du Sein par Intelligence Artificielle

![Logo UM5](images/logo_um5.png)

---

## Description

Ce projet consiste en une application web développée avec **Streamlit** pour la détection automatisée du cancer du sein à partir d’images échographiques. Le modèle utilisé est un **MobileNetV2 fine-tuné** sur le dataset BUSI, capable de classer les images en trois catégories : **Bénin**, **Malin**, et **Normal**.

L’objectif est de fournir un outil d’aide au diagnostic rapide et accessible, tout en rappelant que ce système ne remplace pas un diagnostic médical professionnel.

---

## Fonctionnalités

- Chargement d’images échographiques au format JPG, JPEG ou PNG.
- Prédiction instantanée de la classe de l’image avec affichage de la confiance.
- Visualisation graphique des probabilités pour chaque classe.
- Recommandations basées sur le résultat de la prédiction.
- Interface utilisateur claire et esthétique avec fond personnalisé et logo UM5.

---

## Installation

1. Cloner ce dépôt :
   ```bash
   git clone https://github.com/ton-utilisateur/ton-projet.git
   cd ton-projet
   ```

2. Créer un environnement virtuel (optionnel mais recommandé) :

conda create -n breast-cancer-detector python=3.10
conda activate breast-cancer-detector

3. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

4. Assure-toi que le modèle fine-tuné est placé dans le dossier `models/` sous le nom `fine_tuned_model.keras`.

---

## Usage

Lancer l’application Streamlit avec la commande :

```bash
streamlit run app.py
```

Ouvre ensuite l’URL affichée dans ton navigateur pour accéder à l’interface.

---

## Structure du projet

```
/images                # Images pour le fond et le logo
/models                # Modèle TensorFlow fine-tuné (.keras)
/app.py                # Script principal Streamlit
/requirements.txt      # Librairies Python nécessaires
/README.md             # Ce fichier
```

---

## Résultats et performances

✅ Rapport de classification (jeu de test strict)

Classe	Précision	Rappel	F1-score	Support
Bénin	0.94	0.93	0.94	179
Malin	0.84	0.90	0.87	84
Normal	0.96	0.91	0.93	53
Exactitude globale			0.92	316

🧪 Résultats globaux sur l’ensemble de test complet
Accuracy globale : 91%

Loss : 0.2368

📊 Matrice de confusion brute :

lua
Copier
Modifier
[[840  44   7]
 [ 52 364   5]
 [ 28  10 228]]
---

## Limites connues

- Le modèle peut confondre certaines images normales avec des images malignes (faux positifs).
- L’outil doit être utilisé uniquement comme support à la décision médicale.
- Les performances peuvent varier en fonction de la qualité et des caractéristiques des images échographiques.

---

## Avertissement médical

Cette application est fournie à titre informatif uniquement. Elle ne remplace pas un diagnostic médical professionnel. Consultez toujours un spécialiste qualifié pour toute question médicale.

---

## Auteur

Souad ABOUD & Abderrazak NADIR 
Enadré et Supervisé par Pr. Abdelhak MAHMOUDI
Master IT / Projet Deep Learning  
Université Mohammed V - Rabat  

---

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus d’informations.
