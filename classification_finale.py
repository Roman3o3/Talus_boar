import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Lecture des données
sangliers = pd.read_excel("C:/Users/Bordeaux Montaigne/Desktop/Article/Tableau_actuel.xlsx")

# Filtrer et regrouper les contextes
contexte_mapping = {
    'pen': 'Captive',
    'stall': 'Captive',
    'control': 'Wild',
    'hunted': 'Wild'
}

sangliers_mesolithic = sangliers[sangliers['Context'].isin(['Mesolithic'])]
sangliers = sangliers[~sangliers['Context'].isin(['Mesolithic'])]
sangliers['Context'] = sangliers['Context'].map(contexte_mapping)
sangliers = sangliers.dropna(subset=['Context'])

# Sélection des caractéristiques et de la cible
X = sangliers[['C', '%Trab', 'TC', 'RMeanT', 'RMaxT']]
y = sangliers['Context']

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', SVC(probability=True))
])

# Grille d'hyperparamètres étendue
params = {
    'classifier__C': np.logspace(-2, 2, 5),  # [0.01, 0.1, 1, 10, 100]
    'classifier__kernel': ['linear', 'rbf', 'poly'],
    'classifier__gamma': ['scale', 'auto'] + list(np.logspace(-2, 1, 4)),
    'classifier__degree': [2, 3]  # Pour le noyau poly
}

# Recherche des hyperparamètres
grid_search = GridSearchCV(pipeline, params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Analyse des résultats
print("\n=== MEILLEURS HYPERPARAMÈTRES ===")
print("Paramètres optimaux:", grid_search.best_params_)
print(f"Score moyen en validation croisée: {grid_search.best_score_:.3f}")

# Détails sur les paramètres
best = grid_search.best_estimator_.named_steps['classifier']
print("\n=== INTERPRÉTATION DES PARAMÈTRES ===")
print(f"- Noyau sélectionné: {best.kernel}")
print(f"- C (force de régularisation): {best.C:.3f}")
print(f"- Gamma (influence des points): {best.gamma}")

if best.kernel == 'poly':
    print(f"- Degré du polynôme: {best.degree}")

# Évaluation finale
y_pred = grid_search.best_estimator_.predict(X_test)
print("\n=== PERFORMANCE FINALE ===")
print(f"Accuracy sur le test: {accuracy_score(y_test, y_pred):.3f}")
print("Rapport de classification:")
print(classification_report(y_test, y_pred))

# Prédiction des Mésolithiques
if not sangliers_mesolithic.empty:
    X_mesolithic = sangliers_mesolithic[['C', '%Trab', 'TC', 'RMeanT', 'RMaxT']]
    probas = grid_search.best_estimator_.predict_proba(X_mesolithic)

    sangliers_mesolithic['Predicted_Context'] = grid_search.best_estimator_.predict(X_mesolithic)
    sangliers_mesolithic['Prob_Captive'] = probas[:, 0]
    sangliers_mesolithic['Prob_Wild'] = probas[:, 1]

    print("\n=== PRÉDICTIONS MÉSOLITHIQUES ===")
    print(sangliers_mesolithic[['ID', 'Predicted_Context', 'Prob_Captive', 'Prob_Wild']].to_string(index=False))

# Export des résultats
results_df = pd.DataFrame(grid_search.cv_results_)
results_df.to_excel("svm_hyperparameter_results.xlsx", index=False)