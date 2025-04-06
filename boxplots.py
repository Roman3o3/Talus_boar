import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Charger les données
df = pd.read_excel("C:/Users/Bordeaux Montaigne/Desktop/Article/Tableau_actuel.xlsx")

# Sélectionner les colonnes
colonnes_selectionnees = ['Context', 'WBV (cm³)', 'C', '%Trab', 'TC', 'RMeanT', 'RMaxT']
df_selected = df[colonnes_selectionnees]

# Créer une figure avec 6 sous-graphiques
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()  # Aplatir la matrice d'axes pour faciliter l'itération

# Définir la palette cividis
palette = plt.cm.cividis

# Obtenir les groupes uniques dans 'Context'
groupes = df_selected['Context'].unique()
n_groupes = len(groupes)

# Boucle pour créer les boxplots avec les points superposés
for i, col in enumerate(colonnes_selectionnees[1:]):  # On commence à 1 pour ignorer 'Context'
    # Boxplot
    sns.boxplot(x='Context', y=col, data=df_selected, ax=axes[i], palette=palette(np.linspace(0, 1, n_groupes)))

    # Swarmplot avec les couleurs de la palette cividis
    for j, groupe in enumerate(groupes):
        mask = (df_selected['Context'] == groupe)
        couleur = palette(j / n_groupes)  # Couleur correspondante dans la palette
        sns.swarmplot(x=df_selected.loc[mask, 'Context'], y=df_selected.loc[mask, col],
                      color=couleur, ax=axes[i], alpha=0.7, label=groupe if i == 0 else "")

    # Supprimer le titre du graphique
    axes[i].set_title("")

    # Nommer les axes x et y
    axes[i].set_xlabel('Context')
    axes[i].set_ylabel(col)

    # Ajouter une légende uniquement pour le premier graphique
    if i == 0:
        axes[i].legend(title='Context', bbox_to_anchor=(1.05, 1), loc='upper left')  # Déplacer la légende à droite

# Ajuster l'espacement entre les sous-graphiques
plt.tight_layout()

# Afficher la figure
plt.show()