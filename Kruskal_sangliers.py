import pandas as pd
from scipy import stats
from scikit_posthocs import posthoc_dunn
import numpy as np
import openpyxl
from itertools import combinations


# Fonction pour calculer la taille d'effet pour Dunn
def compute_dunn_effect_size(group1, group2):
    n1, n2 = len(group1), len(group2)
    ranks = stats.rankdata(np.concatenate([group1, group2]))
    r1 = np.mean(ranks[:n1])
    r2 = np.mean(ranks[n1:])
    return (r2 - r1) / np.sqrt((n1 + n2 + 1) * (1 / n1 + 1 / n2) / 12)


# Charger les données
sangliers = pd.read_excel("C:/Users/Bordeaux Montaigne/Desktop/Article/Tableau_actuel.xlsx")

# Variables à analyser
variables_y = ['WBV (cm³)', 'C', '%Trab', 'TC', 'RMeanT', 'RMaxT']
variable_x = 'Context'

# Créer un fichier Excel pour les résultats
writer = pd.ExcelWriter('Resultats_Kruskal_Dunn_ameliore.xlsx', engine='openpyxl')

# Dictionnaire pour stocker tous les résultats
all_results = {}

# Boucle pour chaque variable Y
for var in variables_y:
    # Préparer les données (supprimer les NaN)
    data_clean = sangliers[[variable_x, var]].dropna()
    groups = data_clean.groupby(variable_x)[var].apply(list)
    group_names = list(groups.keys())

    # Test de Kruskal-Wallis
    h_stat, p_val = stats.kruskal(*groups)
    dof = len(groups) - 1  # degrés de liberté

    # Stocker les résultats
    results = {
        'Kruskal_H': h_stat,
        'Kruskal_dof': dof,
        'Kruskal_p': p_val,
        'Dunn_test': None
    }

    # Si significatif, faire le test de Dunn complet avec Z et tailles d'effet
    if p_val < 0.05:
        # Test de Dunn standard (pour p-values ajustées)
        dunn_result = posthoc_dunn(data_clean, val_col=var, group_col=variable_x, p_adjust='bonferroni')

        # Créer une liste pour stocker les résultats
        dunn_data = []

        for (g1, g2) in combinations(group_names, 2):
            # Score Z (approximation)
            n1, n2 = len(groups[g1]), len(groups[g2])
            N = n1 + n2
            H = h_stat
            Z = np.sqrt((12 * H) / (N * (N + 1)) * (1 / n1 + 1 / n2))

            # Taille d'effet
            eff_size = compute_dunn_effect_size(groups[g1], groups[g2])

            # Récupérer p-value ajustée
            p_adj = dunn_result.loc[g1, g2]

            dunn_data.append({
                'Groupe1': g1,
                'Groupe2': g2,
                'Z': Z,
                'p_ajust': p_adj,
                'Taille_effet': eff_size
            })

        # Convertir en DataFrame
        dunn_extended = pd.DataFrame(dunn_data)
        results['Dunn_test'] = dunn_extended
        dunn_extended.to_excel(writer, sheet_name=f'Dunn_{var[:20]}', index=False)

    all_results[var] = results

# Créer un DataFrame pour les résultats Kruskal-Wallis
kruskal_results = pd.DataFrame.from_dict(all_results, orient='index')
kruskal_results.to_excel(writer, sheet_name='Kruskal_Wallis')

# Sauvegarder le fichier Excel
writer.close()

print("Analyse terminée. Les résultats complets ont été sauvegardés dans 'Resultats_Kruskal_Dunn_ameliore.xlsx'")