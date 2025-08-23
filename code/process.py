import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

data_path = r"C:\Users\Admin\Documents\VScode\math project\bmi\data\normalized_data.json"
output_path = r"C:\Users\Admin\Documents\VScode\math project\bmi\data\bias_and_coef.json"

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    X = []
    y = []
    feature_names =['sex','active_intensity', 'sleep_per_week', 'sumw']
    class_map = {"Underweight": 0, "Normal": 1, "Overweight": 2}
    
    for item in data:
        X.append([item[feature] for feature in feature_names])
        y.append(class_map[item['class']])
    
    return np.array(X), np.array(y), class_map, feature_names

def print_confusion_matrix(cm, labels):
    print("\nConfusion Matrix:")
    header = "\t" + "\t".join(labels)
    print(header)
    for i, row in enumerate(cm):
        row_label = labels[i]
        row_values = "\t".join(str(x) for x in row)
        print(f"{row_label}\t{row_values}")

def train_and_evaluate(X, y, fraction, run_id, feature_names):
    size = int(len(X) * fraction)
    X_sub, y_sub = X[:size], y[:size]
    X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)

    model = LogisticRegression(solver='lbfgs', max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== Training with {int(fraction*100)}% of data ===")
    print(f"Accuracy: {acc:.4f}")
    labels = ["Underweight", "Normal", "Overweight"]
    print_confusion_matrix(cm, labels)

    output = {
        "features": feature_names,
        "bias": model.intercept_.tolist(),
        "coef": model.coef_.tolist()
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    return model

if __name__ == "__main__":
    X, y, class_map, feature_names = load_data(data_path)
    fractions = [0.25, 0.5, 0.75, 1.0]

    for i, frac in enumerate(fractions):
        train_and_evaluate(X, y, frac, i+1, feature_names)

    sample_input = {
        "sex": 1,                
        "age": 1.2,              
        "weight": 0.1,           
        "height": 0.3,           
        "active_intensity": -0.5,
        "sleep_per_week": 0.2,   
        "sumw": -1.0             
    }

    normalized = [sample_input[feature] for feature in feature_names]

    print("\n=== Sample Prediction ===")
    model = train_and_evaluate(X, y, 1.0, "final", feature_names)
    pred_class = model.predict([normalized])[0]
    inv_class_map = {v: k for k, v in class_map.items()}
    print("Predicted class:", inv_class_map[pred_class])

