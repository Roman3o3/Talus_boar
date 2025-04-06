import pandas as pd
from scipy.stats import mannwhitneyu
from openpyxl import Workbook

# Charger les données
data = pd.read_excel("C:/Users/Bordeaux Montaigne/Desktop/Article/Tableau_actuel.xlsx")

# Variables à analyser
variables = ['Mass (kg)', 'WBV (cm³)']
group_var = 'Sex'
context_var = 'Context'

# Nettoyage des données
data_clean = data[['Context', 'Sex'] + variables].dropna()

# Initialisation du fichier Excel
wb = Workbook()
ws = wb.active
ws.title = "Résultats_Wilcoxon"

# En-tête du tableau
ws.append(["Test de Wilcoxon-Mann-Whitney (par contexte)"])
ws.append([])
headers = ["Contexte", "Variable", "Groupe1", "Groupe2",
           "n1", "n2", "Statistique U", "p-value", "Différence significative"]
ws.append(headers)

# Boucle d'analyse
for context in data_clean[context_var].unique():
    context_data = data_clean[data_clean[context_var] == context]

    # Vérifier qu'il y a bien 2 sexes dans ce contexte
    if len(context_data[group_var].unique()) != 2:
        continue

    group1, group2 = context_data[group_var].unique()

    for var in variables:
        # Séparation des groupes
        sample1 = context_data[context_data[group_var] == group1][var]
        sample2 = context_data[context_data[group_var] == group2][var]

        # Test de Wilcoxon-Mann-Whitney
        try:
            U, p = mannwhitneyu(sample1, sample2, alternative='two-sided')

            # Ajout des résultats
            ws.append([
                context,
                var,
                group1,
                group2,
                len(sample1),
                len(sample2),
                U,
                p,
                "Oui" if p < 0.05 else "Non"
            ])
        except:
            ws.append([context, var, "Erreur", "", "", "", "", ""])

# Ajustement automatique des colonnes
for col in ws.columns:
    max_length = 0
    column = col[0].column_letter
    for cell in col:
        try:
            if len(str(cell.value)) > max_length:
                max_length = len(str(cell.value))
        except:
            pass
    adjusted_width = (max_length + 2) * 1.2
    ws.column_dimensions[column].width = adjusted_width

# Sauvegarde
output_path = "C:/Users/Bordeaux Montaigne/Desktop/PycharmProjects/30DaysOfPython/Tests_Wilcoxon_par_Contexte.xlsx"
wb.save(output_path)
print(f"Résultats sauvegardés dans : {output_path}")