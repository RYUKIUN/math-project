import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# === File paths ===
data_path = r"C:\Users\Admin\Documents\VScode\math project\new refined\data\normalized_data.json"

# === Load dataset ===
def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    X = []
    # Updated feature names to match new normalized data
    feature_names = ['sex', 'age', 'weight', 'height', 'active_intensity', 'sleep_per_week', 'sumw']
    for item in data:
        X.append([
            item['sex'],
            item['age'],
            item['weight'],
            item['height'],
            item['active_intensity'],
            item['sleep_per_week'],
            item['sumw']
        ])
    return pd.DataFrame(X, columns=feature_names)

# === Calculate and display correlation matrix ===
def compute_correlation(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    denominator_y = sum((y[i] - mean_y) ** 2 for i in range(n))
    
    denominator = (denominator_x * denominator_y) ** 0.5
    if denominator == 0:
        return 0  # Avoid division by zero
    return numerator / denominator

def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=[np.number])
    columns = numeric_df.columns
    n = len(columns)
    
    corr_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            x = numeric_df.iloc[:, i].tolist()
            y = numeric_df.iloc[:, j].tolist()
            corr_matrix[i, j] = compute_correlation(x, y)
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(n), columns, rotation=45, ha='right')
    plt.yticks(range(n), columns)
    
    # Add text annotations
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{corr_matrix[i, j]:.2f}", ha='center', va='center', color='black')
    
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

# === Calculate VIF ===
def calculate_vif(df):
    df_with_const = df.copy()
    df_with_const['const'] = 1  # Add constant for intercept
    vif_data = pd.DataFrame()
    vif_data['Feature'] = df.columns
    vif_data['VIF'] = [variance_inflation_factor(df_with_const.values, i)
                       for i in range(len(df.columns))]
    return vif_data

# === Main ===
if __name__ == "__main__":
    df = load_data(data_path)

    print("\n=== Correlation Matrix ===")
    print(df.corr())

    plot_correlation_matrix(df)

    print("\n=== Variance Inflation Factor (VIF) ===")
    vif_result = calculate_vif(df)
    print(vif_result)
