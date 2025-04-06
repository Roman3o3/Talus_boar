import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Charger les données
sangliers = pd.read_excel("C:/Users/Bordeaux Montaigne/Desktop/Article/Tableau_actuel.xlsx")

# Préparation des données
df_clean = sangliers.copy()

# Convertir le sexe en numérique (M=1, F=0)
df_clean['Sex'] = df_clean['Sex'].map({'M': 1, 'F': 0})

# Supprimer les unités entre parenthèses
df_clean.columns = [col.split(' (')[0] for col in df_clean.columns]

# Sélectionner les colonnes numériques
numeric_cols = ['Sex', 'Age', 'Mass', 'WBV', 'C', '%Trab', 'TC', 'RMeanT', 'RMaxT']
df_numeric = df_clean[numeric_cols].select_dtypes(include=['number'])


# Fonction pour calculer Pearson et p-values
def calculate_pearson_pvalues(df):
    cols = df.columns
    n = len(cols)
    corr_matrix = np.zeros((n, n))
    pvalue_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Sélectionner les paires valides (non-NA)
            valid_idx = df[[cols[i], cols[j]]].dropna().index
            if len(valid_idx) >= 3:  # Minimum 3 observations
                x = df.loc[valid_idx, cols[i]]
                y = df.loc[valid_idx, cols[j]]
                corr, pvalue = pearsonr(x, y)
                corr_matrix[i, j] = corr
                pvalue_matrix[i, j] = pvalue
            else:
                corr_matrix[i, j] = np.nan
                pvalue_matrix[i, j] = np.nan

    return corr_matrix, pvalue_matrix


# Calculer les matrices
corr_matrix, pvalue_matrix = calculate_pearson_pvalues(df_numeric)

# Créer les DataFrames
corr_df = pd.DataFrame(corr_matrix,
                       index=df_numeric.columns,
                       columns=df_numeric.columns)
pvalue_df = pd.DataFrame(pvalue_matrix,
                         index=df_numeric.columns,
                         columns=df_numeric.columns)

# Créer le masque triangulaire supérieur (pour afficher seulement le triangle inférieur)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Créer le texte d'annotation combiné
annot_text = np.empty_like(corr_matrix, dtype='U50')
for i in range(corr_matrix.shape[0]):
    for j in range(corr_matrix.shape[1]):
        if not np.isnan(corr_matrix[i, j]) and not mask[i, j]:  # Seulement le triangle inférieur
            # Formater corrélation et p-value
            corr_str = f"r = {corr_matrix[i, j]:.2f}"

            if pvalue_matrix[i, j] < 0.0001:
                p_str = "p < 0.0001"
            else:
                p_str = f"p = {pvalue_matrix[i, j]:.4f}"

            # Ajouter les étoiles de significativité
            stars = ""
            if pvalue_matrix[i, j] < 0.001:
                stars = "***"
            elif pvalue_matrix[i, j] < 0.01:
                stars = "**"
            elif pvalue_matrix[i, j] < 0.05:
                stars = "*"

            annot_text[i, j] = f"{corr_str}\n{p_str}\n{stars}"
        else:
            annot_text[i, j] = ""

# Paramètres de visualisation
plt.figure(figsize=(12, 10))

# Créer la heatmap
heatmap = sns.heatmap(corr_df,
                      annot=annot_text,
                      fmt="",
                      cmap='coolwarm',
                      vmin=-1,
                      vmax=1,
                      mask=mask,  # Masquer le triangle supérieur
                      cbar_kws={'label': 'Coefficient r de Pearson'},
                      annot_kws={'size': 9,
                                 'ha': 'center',
                                 'va': 'center'})

# Personnalisation des axes
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# Ajuster les marges
plt.tight_layout()

# Sauvegarder en haute qualité
plt.savefig("matrice_pearson_triangle_inf.png",
            dpi=300,
            bbox_inches='tight',
            transparent=False)

# Afficher le graphique
plt.show()

# Exporter les données brutes vers Excel
with pd.ExcelWriter('resultats_pearson.xlsx') as writer:
    corr_df.to_excel(writer, sheet_name='Pearson_r')
    pvalue_df.to_excel(writer, sheet_name='p_values')