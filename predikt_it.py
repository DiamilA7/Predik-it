import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Chargement des données
df = pd.read_csv("StudentPerformanceFactors.csv")   

# Conversion en /20
df['Exam_Score_20'] = df['Exam_Score'] / 5
df['Previous_Scores_20'] = df['Previous_Scores'] / 5

print("Dataset chargé :", df.shape)
target = 'Exam_Score_20'

# Features
num_features = ['Hours_Studied', 'Sleep_Hours', 'Attendance', 'Previous_Scores_20', 'Physical_Activity']
cat_features = ['Motivation_Level', 'School_Type', 
                'Peer_Influence', 'Learning_Disabilities', 'Parental_Involvement']

X = df[num_features + cat_features]
y = df[target]

# Pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', KNeighborsRegressor(n_neighbors=12, weights='distance'))
])

# Entraînement
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("─── RÉSULTATS ───────────────────────")
print(f"RMSE  : {rmse:.2f} points")
print(f"R²  qualité du model  : {r2:.3f}")
print("─────────────────────────────────────\n")

# Mapping Français → Anglais
mapping = {
    "motivation": {"bas": "Low", "moyen": "Medium", "élevé": "High", "faible": "Low", "fort": "High"},
    "ecole": {"publique": "Public", "privée": "Private"},
    "influence": {"positive": "Positive", "positif": "Positive", "neutre": "Neutral", "négative": "Negative", "negatif": "Negative"},
    "troubles": {"oui": "Yes", "non": "No"},
    "parents": {"bas": "Low", "moyen": "Medium", "élevé": "High"}
}

# Fonction principale
def predikt_it():
    print("=== PREDIKT-IT - Moyenne Générale ===\n")
    
    data = {}
    
    data['Hours_Studied']      = float(input("Heures d'étude par semaine ? "))
    data['Sleep_Hours']        = float(input("Heures de sommeil par nuit ? "))
    data['Attendance']         = float(input("Taux de présence (%) ? "))
    data['Previous_Scores_20'] = float(input("Note précédente (/20) ? "))
    data['Physical_Activity']  = float(input("Heures d'activité physique par semaine ? "))
    
    # Catégoriels
    mot = input("Niveau de motivation (Bas/Moyen/Élevé) ? ").strip().lower()
    data['Motivation_Level'] = mapping["motivation"].get(mot, "Medium")
    
    eco = input("Type d'école (Publique/Privée) ? ").strip().lower()
    data['School_Type'] = mapping["ecole"].get(eco, "Public")
    
    inf = input("Influence des camarades (Positive/Neutre/Négative) ? ").strip().lower()
    data['Peer_Influence'] = mapping["influence"].get(inf, "Neutral")
    
    trou = input("Troubles d'apprentissage (Oui/Non) ? ").strip().lower()
    data['Learning_Disabilities'] = mapping["troubles"].get(trou, "No")
    
    par = input("Implication des parents (Bas/Moyen/Élevé) ? ").strip().lower()
    data['Parental_Involvement'] = mapping["parents"].get(par, "Medium")
    
    # Prédiction
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]
    
    print(f"\n→ Moyenne générale prédite : {prediction:.2f} / 20\n")


if __name__ == "__main__":
    predikt_it()

# Sauvegarde
import joblib
joblib.dump(model, 'predikt_it_model_20.pkl')
print("Modèle sauvegardé.")
