import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull

df = pd.read_excel("C:/Users/Bordeaux Montaigne/Desktop/Article/Tableau_actuel.xlsx")

# Sélectionner les colonnes H à L (index 7 à 11 en Python, car l'indexation commence à 0)
variables = df.iloc[:, 7:12]

# Convertir la colonne B en codes numériques pour la coloration
couleurs, noms_groupes = pd.factorize(df.iloc[:, 1])

# Étape 2: Standardiser les données
scaler = StandardScaler()
variables_standardisees = scaler.fit_transform(variables)

# Étape 3: Appliquer l'ACP
pca = PCA(n_components=2)  # Réduire à 2 composantes principales pour la visualisation
composantes_principales = pca.fit_transform(variables_standardisees)

# Calculer les pourcentages d'inertie expliquée par chaque axe
pourcentage_explique = pca.explained_variance_ratio_ * 100

# Étape 4: Visualiser les résultats de l'ACP (sans quadrillage)
plt.figure(figsize=(10, 8))

# Utiliser une palette adaptée aux daltoniens (cividis)
palette = plt.cm.cividis

# Tracer les points pour chaque groupe
for groupe in np.unique(couleurs):
    # Sélectionner les points du groupe
    mask = (couleurs == groupe)
    couleur = palette(groupe / len(noms_groupes))  # Même couleur pour les points et le polygone
    plt.scatter(composantes_principales[mask, 0], composantes_principales[mask, 1],
                color=couleur, label=noms_groupes[groupe], alpha=0.7)

    # Tracer le polygone (enveloppe convexe) pour le groupe
    if np.sum(mask) >= 3:  # Il faut au moins 3 points pour former un polygone
        hull = ConvexHull(composantes_principales[mask])
        for simplex in hull.simplices:
            plt.plot(composantes_principales[mask][simplex, 0], composantes_principales[mask][simplex, 1],
                     color=couleur, linestyle='--', alpha=0.5)
        # Remplir l'intérieur du polygone avec une transparence
        plt.fill(composantes_principales[mask][hull.vertices, 0], composantes_principales[mask][hull.vertices, 1],
                 color=couleur, alpha=0.1)  # Transparence légère pour l'intérieur

# Ajouter les pourcentages expliqués par chaque axe
plt.xlabel(f'PC1 ({pourcentage_explique[0]:.1f}%)')
plt.ylabel(f'PC2 ({pourcentage_explique[1]:.1f}%)')

# Titre et légende
plt.legend(title='Groups', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(False)  # Désactiver le quadrillage
plt.tight_layout()
plt.show()

# Étape 5: Tracer le cercle des corrélations dans un deuxième graphique
plt.figure(figsize=(8, 8))

# Calculer les corrélations entre les variables et les composantes principales
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Tracer le cercle des corrélations
for i, (x, y) in enumerate(loadings):
    plt.arrow(0, 0, x, y, color='black', alpha=0.5, head_width=0.05)
    plt.text(x * 1.2, y * 1.2, variables.columns[i], color='black', ha='center', va='center')

# Ajouter un cercle unité
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
plt.gca().add_artist(circle)

# Ajuster les limites du graphique
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)

# Ajouter les labels et le titre
plt.xlabel(f'PC1 ({pourcentage_explique[0]:.1f}%)')
plt.ylabel(f'PC2 ({pourcentage_explique[1]:.1f}%)')
plt.grid(False)  # Désactiver le quadrillage
plt.tight_layout()
plt.show()