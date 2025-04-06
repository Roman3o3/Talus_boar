import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from openpyxl import Workbook


def permanova_manual(distance_matrix, groups, permutations=999):
    """PerMANOVA manuelle avec test de permutation"""
    # Calcul des sommes de carrés
    SS_total = np.sum(distance_matrix ** 2) / (2 * len(distance_matrix))

    unique_groups = np.unique(groups)
    SS_within = 0
    for g in unique_groups:
        group_indices = np.where(groups == g)[0]
        subgroup_matrix = distance_matrix[np.ix_(group_indices, group_indices)]
        SS_within += np.sum(subgroup_matrix ** 2) / (2 * len(group_indices))

    SS_between = SS_total - SS_within

    # Degrés de liberté
    df_between = len(unique_groups) - 1
    df_within = len(groups) - len(unique_groups)

    # Statistique F
    F = (SS_between / df_between) / (SS_within / df_within)

    # Permutations
    perm_F = []
    for _ in range(permutations):
        perm_groups = np.random.permutation(groups)
        perm_SS_within = 0
        for g in unique_groups:
            group_indices = np.where(perm_groups == g)[0]
            subgroup_matrix = distance_matrix[np.ix_(group_indices, group_indices)]
            perm_SS_within += np.sum(subgroup_matrix ** 2) / (2 * len(group_indices))
        perm_SS_between = SS_total - perm_SS_within
        perm_F.append((perm_SS_between / df_between) / (perm_SS_within / df_within))

    p_value = (np.sum(perm_F >= F) + 1) / (permutations + 1)
    R2 = SS_between / SS_total

    return {
        'R2': R2,
        'F': F,
        'p_value': p_value,
        'df_between': df_between,
        'df_within': df_within,
        'n_groups': len(unique_groups),
        'n_permutations': permutations
    }


# 1. Chargement des données
data = pd.read_excel("C:/Users/Bordeaux Montaigne/Desktop/Article/Tableau_actuel.xlsx")
variables_y = ['C', '%Trab', 'TC', 'RMeanT', 'RMaxT']
variable_x = 'Sex'  # Changed from 'Context' to 'Sex'
data_clean = data[[variable_x] + variables_y].dropna()

# 2. Calcul de la matrice de distance
data_std = (data_clean[variables_y] - data_clean[variables_y].mean()) / data_clean[variables_y].std()
distance_matrix = squareform(pdist(data_std, 'euclidean'))

# 3. Exécution de la PerMANOVA
results = permanova_manual(distance_matrix, data_clean[variable_x].values)

# 4. Création du fichier Excel
wb = Workbook()
ws = wb.active
ws.title = "PerMANOVA_Results"

# Écriture des résultats
ws.append(["PerMANOVA Results"])
ws.append([f"Variable explicative (X): {variable_x}"])
ws.append([f"Variables réponse (Y): {', '.join(variables_y)}"])
ws.append([])
ws.append(["Statistique", "Valeur"])
ws.append(["R² (variance expliquée)", results['R2']])
ws.append(["F-statistique", results['F']])
ws.append(["p-value", results['p_value']])
ws.append(["Degrés de liberté (groupes)", results['df_between']])
ws.append(["Degrés de liberté (résidus)", results['df_within']])
ws.append(["Nombre de groupes", results['n_groups']])
ws.append(["Nombre de permutations", results['n_permutations']])
ws.append(["Nombre d'observations", len(data_clean)])

# 5. Sauvegarde avec un nouveau nom pour éviter l'écrasement
output_path = "PerMANOVA_Results_Sex.xlsx"  # Changed filename
wb.save(output_path)
print(f"Résultats sauvegardés dans : {output_path}")