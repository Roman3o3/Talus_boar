import pandas as pd

# Charger les données
sangliers = pd.read_excel("C:/Users/Bordeaux Montaigne/Desktop/Article/Tableau_actuel.xlsx")
print(sangliers['C'].describe())
print(sangliers['TC'].describe())