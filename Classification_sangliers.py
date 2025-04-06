import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Lecture et nettoyage des données
sangliers = pd.read_excel("C:/Users/Bordeaux Montaigne/Desktop/Article/Tableau_actuel.xlsx")

# Filtrer et regrouper les contextes
contexte_mapping = {
    'pen': 'Captive',
    'stall': 'Captive',
    'control': 'Wild',
    'hunted': 'Wild'
}

sangliers = sangliers[~sangliers['Context'].isin(['Mesolithic'])]
sangliers['Context'] = sangliers['Context'].map(contexte_mapping)
sangliers = sangliers.dropna(subset=['Context'])

# Sélection des caractéristiques et de la cible
X = sangliers[['C', '%Trab', 'TC', 'RMeanT', 'RMaxT']]
y = sangliers['Context']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Création d'un pipeline pour le prétraitement et la classification
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier())  # Classificateur par défaut
])

# Définition des modèles et des hyperparamètres à tester
models = {
    'KNN': {
        'classifier': [KNeighborsClassifier()],
        'classifier__n_neighbors': [3, 5, 7, 9],
        'classifier__weights': ['uniform', 'distance']
    },
    'RandomForest': {
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]
    },
    'SVM': {
        'classifier': [SVC(probability=True)],
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf']
    },
    'GradientBoosting': {
        'classifier': [GradientBoostingClassifier()],
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1, 1],
        'classifier__max_depth': [3, 5, 7]
    }
}

# Recherche des meilleurs hyperparamètres pour chaque modèle
best_models = {}
for model_name, params in models.items():
    print(f"\nRecherche des meilleurs hyperparamètres pour {model_name}")
    grid_search = GridSearchCV(pipeline, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_
    print(f"Meilleurs paramètres: {grid_search.best_params_}")
    print(f"Score de validation croisée: {grid_search.best_score_:.4f}")

# Évaluation des modèles sur l'ensemble de test
print("\nRésultats sur l'ensemble de test:")
for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nPerformance de {model_name}:")
    print(f"Précision: {accuracy:.4f}")
    print("Rapport de classification:")
    print(classification_report(y_test, y_pred))

# Sélection du meilleur modèle
best_model_name = max(best_models, key=lambda k: accuracy_score(y_test, best_models[k].predict(X_test)))
print(f"\nLe meilleur modèle est {best_model_name} avec une précision de {accuracy_score(y_test, best_models[best_model_name].predict(X_test)):.4f}")