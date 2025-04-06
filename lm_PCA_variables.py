import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from openpyxl import Workbook

# Load and prepare data
df = pd.read_excel("C:/Users/Bordeaux Montaigne/Desktop/Article/Tableau_actuel.xlsx")
variables = df.iloc[:, 7:12]  # RMeanT, RMaxT, C, TC, %Trab
predictors = df[['Mass (kg)', 'WBV (cm³)', 'Age (months)']].dropna()

# Standardize and perform PCA
scaler = StandardScaler()
variables_std = scaler.fit_transform(variables)
pca = PCA(n_components=2)
components = pca.fit_transform(variables_std)

# Prepare results table
results = []

# Perform regressions for each predictor on each PC
for i, pc in enumerate(['PC1', 'PC2']):
    for predictor in predictors.columns:
        # Merge data ensuring alignment
        merged = pd.concat([pd.Series(components[:, i], name=pc), predictors[predictor]], axis=1).dropna()

        # Linear regression
        X = sm.add_constant(merged[predictor])
        y = merged[pc]
        model = sm.OLS(y, X).fit()

        # Store results
        results.append({
            'Dependent': pc,
            'Independent': predictor,
            'R²': model.rsquared,
            'Adj. R²': model.rsquared_adj,
            'Coefficient': model.params[1],
            'Std Error': model.bse[1],
            't-value': model.tvalues[1],
            'p-value': model.pvalues[1],
            'N': len(merged)
        })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Create Excel output
wb = Workbook()
ws = wb.active
ws.title = "Regression Results"

# Write headers
headers = ["Dependent", "Independent", "R²", "Adj. R²", "Coefficient",
           "Std Error", "t-value", "p-value", "N"]
ws.append(headers)

# Write data
for _, row in results_df.iterrows():
    ws.append([
        row['Dependent'],
        row['Independent'],
        f"{row['R²']:.3f}",
        f"{row['Adj. R²']:.3f}",
        f"{row['Coefficient']:.3f}",
        f"{row['Std Error']:.3f}",
        f"{row['t-value']:.3f}",
        f"{row['p-value']:.4f}",
        row['N']
    ])

# Format columns
for col in ws.columns:
    max_length = max(len(str(cell.value)) for cell in col)
    ws.column_dimensions[col[0].column_letter].width = max_length + 2

# Save
output_path = "PCA_Regression_Results.xlsx"
wb.save(output_path)
print(f"Results saved to {output_path}")

# Display results
print("\nRegression Results:")
print(results_df[['Dependent', 'Independent', 'R²', 'p-value', 'Coefficient']].to_string(index=False))