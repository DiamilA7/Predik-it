# PrediktIt

PrediktIt est un programme qui permet de prédire la moyenne générale d’un étudiant sur 20.

Il a été développé à partir d’un dataset réel contenant plus de 6600 étudiants.

---

## Objectif

Prédire la moyenne générale d’un étudiant en fonction de ses habitudes et de son contexte personnel.

Le modèle utilise les informations suivantes :
- Nombre d’heures d’étude par semaine
- Heures de sommeil par nuit
- Taux de présence aux cours
- Note précédente
- Niveau de motivation
- Implication des parents
- Revenus de la famille
- Influence des camarades
- Troubles d’apprentissage
- Activité physique

---

## Dataset

- Nom : StudentPerformanceFactors.csv
- Nombre d’observations : 6607
- Source : Kaggle
- Note moyenne du dataset : 12.5/20

---

## Technologies utilisées

- Python 3
- pandas, numpy
- scikit-learn (Pipeline, ColumnTransformer, KNeighborsRegressor)



## Installation

1. Place le fichier `StudentPerformanceFactors.csv` dans le dossier du projet.
2. Installe les dépendances :


pip install pandas numpy scikit-learn