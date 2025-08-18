import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

data_path = r"C:\Users\Admin\Documents\VScode\math project\bmi\data\normalized_data.json"

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    X = []
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

def compute_correlation(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    denominator_y = sum((y[i] - mean_y) ** 2 for i in range(n))
    
    denominator = (denominator_x * denominator_y) ** 0.5
    if denominator == 0:
        return 0
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
    
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{corr_matrix[i, j]:.2f}", ha='center', va='center', color='black')
    
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_data(data_path)

    print("\n=== Correlation Matrix ===")
    print(df.corr())

    plot_correlation_matrix(df)

