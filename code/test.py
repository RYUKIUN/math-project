import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter
import math

# File path to the json data
file_path = r"C:\Users\Admin\Documents\VScode\math project\bmi\data\updated_data.json"
acc_record = []

def bmr_and_tdee(weight, high, age, sex, workout):
    match workout.lower():
        case "no": factor = 1.2
        case "light": factor = 1.375
        case "normal": factor = 1.55
        case "active": factor = 1.725
        case "high": factor = 1.9
        case "": print("workout value error")

    if sex == "ช.":
        bmr = 88.362 + (13.397 * weight) + (4.799 * high) - (5.677 * age)
    if sex == "ญ.":
        bmr = 447.593 + (9.247 * weight) + (3.098 * high) - (4.33 * age)
    return bmr

def bri(wrist, heigh):
    if wrist / heigh > 1:
        print("value error : BRI ratio must be less or equal to 1")
    bri = 364.2 - (365.5 * (math.sqrt(1 - ((wrist / heigh) ** 2))))
    return bri

def age_cap(age):
    if age > 18:
        age = 18
    if age < 12:
        age = 12
    return age

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_bmi(weight, height):
    height_m = height / 100
    return round(weight / (height_m ** 2), 2)

def classify_bmi(bmi, age, gender):
    bmi_thresholds = {
        'ช.': {
            12: (21.9, 26.0),
            13: (22.6, 26.8),
            14: (23.3, 27.7),
            15: (23.9, 28.6),
            16: (24.6, 29.5),
            17: (25.1, 30.3),
            18: (25.7, 31.1)
        },
        'ญ.': {
            12: (21.7, 26.4),
            13: (22.4, 27.2),
            14: (23.2, 28.1),
            15: (23.8, 29.1),
            16: (24.5, 30.0),
            17: (25.1, 30.8),
            18: (25.7, 31.6)
        }
    }

    age = age_cap(age)
    age = int(age)
    overweight_threshold, obesity_threshold = bmi_thresholds[gender][age]

    if bmi < 18.5:
        return "Underweight", (bmi / 35) * 100
    elif bmi < overweight_threshold:
        return "Normal weight", (bmi / 35) * 100
    elif bmi < obesity_threshold:
        return "Overweight", (bmi / 35) * 100
    else:
        return "Obese", (bmi / 35) * 100

def prepare_data(students):
    X, y = [], []
    for student in students:
        try:
            weight = float(student["น้ำหนัก"])
            height = float(student["ส่วนสูง"])
            age = int(student["อายุ(ปี)"]) + (int(student["อายุ(เดือน)"]) / 12)
            daily_calories = student.get("เเคลอรี่ต่อวัน", 0)
            sleep_hours = student.get("ชั่วโมงการนอน", 0)
            exercise_minutes = student.get("เวลาออกกำลังกาย(นาที)", 0)
            gender = student.get("เพศ", "")

            bmi = calculate_bmi(weight, height)
            label, bmi_percentage = classify_bmi(bmi, age, gender)

            X.append([bmi, age, height, weight, daily_calories, sleep_hours, exercise_minutes])
            y.append(label)
        except (ValueError, KeyError) as e:
            print(f"Error processing student data: {e}")
    return np.array(X), np.array(y)

def train_model(X_train, y_train, X_test, y_test):
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    best_model = None
    best_accuracy = 0
    for max_iter in [500, 1000, 1500, 2000, 2500, 3000]:
        print(f"\nTraining Logistic Regression Model with max_iter={max_iter}...\n")
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs',
                                   max_iter=max_iter, verbose=0, class_weight='balanced')
        model.fit(X_train, y_train_encoded)

        acc = accuracy_score(y_test_encoded, model.predict(X_test))
        acc_record.append(acc)
        print(f"Accuracy after {max_iter} iterations: {acc:.2%}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model

        if acc >= 1:
            break

    np.set_printoptions(formatter={'float': '{: .5f}'.format})
    print(f"\nFinal Model Accuracy: {best_accuracy:.2%}")
    print("Intercepts:", best_model.intercept_)
    print("Coefficients:", best_model.coef_)

    # Confusion Matrix and Classification Report
    final_preds = best_model.predict(X_test)
    final_labels = encoder.inverse_transform(final_preds)

    print("\nClassification Report:")
    print(classification_report(y_test, final_labels))

    cm = confusion_matrix(y_test, final_labels, labels=encoder.classes_)

    # Plot confusion matrix using matplotlib instead of seaborn
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.YlGnBu)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    # Add labels and numbers to the plot
    tick_marks = np.arange(len(encoder.classes_))
    plt.xticks(tick_marks, encoder.classes_, rotation=45)
    plt.yticks(tick_marks, encoder.classes_)
    
    # Add text annotations to the plot
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    return best_model, encoder, best_model.intercept_, best_model.coef_

def predict_bmi_category(model, encoder, weight, height, age, daily_calories, sleep_hours, exercise_minutes, gender):
    bmi = calculate_bmi(weight, height)
    input_data = np.array([[bmi, age, height, weight, daily_calories, sleep_hours, exercise_minutes]])
    predicted_label_index = model.predict(input_data)[0]
    predicted_label = encoder.inverse_transform([predicted_label_index])[0]
    probabilities = model.predict_proba(input_data)[0]
    confidence = round(np.max(probabilities) * 100, 2)
    return {"BMI": bmi, "Category": predicted_label, "Confidence (%)": confidence}

def plot_predictions(model, encoder, X_test, y_test, x_variable, intercepts, coefficients):
    y_pred = model.predict(X_test)
    y_pred_labels = encoder.inverse_transform(y_pred)
    colors = {'Underweight': 'blue', 'Normal weight': 'green', 'Overweight': 'orange', 'Obese': 'red'}

    x_variable_mapping = {
        'bmi': 0,
        'height': 2,
        'weight': 3,
        'daily_calories': 4,
        'sleep_hours': 5,
        'exercise_minutes': 6
    }

    x_column_index = x_variable_mapping.get(x_variable)
    if x_column_index is None:
        print("Invalid selection for Y-axis variable.")
        return

    plt.figure(figsize=(10, 6))

    for i in range(len(X_test)):
        plt.scatter(X_test[i][1], X_test[i][x_column_index], color=colors[y_pred_labels[i]],
                    label=y_pred_labels[i] if y_pred_labels[i] not in colors else "")

    plt.xlabel('Age (Years)')
    plt.ylabel(x_variable.replace('_', ' ').title())
    plt.title(f"BMI Classification by {x_variable.replace('_', ' ').title()} and Age")

    x_min, x_max = min(X_test[:, 1]), max(X_test[:, 1])
    y_min, y_max = min(X_test[:, x_column_index]), max(X_test[:, x_column_index])

    plt.xlim(x_min - 2, x_max + 2)
    plt.ylim(y_min - 2, y_max + 2)
    y_range = np.linspace(y_min, y_max, 100)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# New function to display test data and prediction results
def display_test_results(model, encoder, X_test, y_test, num_samples=None):
    """
    Display test data alongside predictions and whether they were correct.
    
    Parameters:
    - model: The trained model
    - encoder: LabelEncoder used to transform categories
    - X_test: Test feature data
    - y_test: True test labels
    - num_samples: Number of samples to display (None for all)
    """
    feature_names = ["BMI", "Age", "Height", "Weight", "Daily Calories", "Sleep Hours", "Exercise Minutes"]
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_labels = encoder.inverse_transform(y_pred)
    
    # Get probabilities for confidence
    y_pred_proba = model.predict_proba(X_test)
    confidence_scores = [round(np.max(proba) * 100, 2) for proba in y_pred_proba]
    
    # Create a results table
    results = []
    for i in range(len(X_test)):
        correct = y_test[i] == y_pred_labels[i]
        result = {
            "Sample": i+1,
            "Features": {feature_names[j]: round(X_test[i][j], 2) for j in range(len(feature_names))},
            "True Category": y_test[i],
            "Predicted Category": y_pred_labels[i],
            "Confidence": confidence_scores[i],  # Fixed: Changed from "Confidence (%)" to "Confidence"
            "Correct": "✓" if correct else "✗"
        }
        results.append(result)
    
    # Limit samples if specified
    if num_samples and num_samples < len(results):
        results = results[:num_samples]
    
    # Print results in a formatted table
    print("\n===== TEST DATA AND PREDICTION RESULTS =====")
    print(f"Displaying {len(results)} out of {len(X_test)} test samples")
    print("="*80)
    
    for result in results:
        print(f"Sample #{result['Sample']}:")
        print(f"  Features:")
        for feature, value in result['Features'].items():
            print(f"    {feature}: {value}")
        print(f"  True Category: {result['True Category']}")
        print(f"  Predicted Category: {result['Predicted Category']}")
        print(f"  Confidence: {result['Confidence']}%")  # Fixed: Using the correct key
        print(f"  Correct Prediction: {result['Correct']}")
        print("-"*80)
    
    # Calculate and display summary statistics
    correct_predictions = sum(1 for r in results if r['True Category'] == r['Predicted Category'])
    accuracy = correct_predictions / len(results)
    
    print(f"Summary for displayed samples:")
    print(f"  Correct predictions: {correct_predictions}/{len(results)} ({accuracy:.2%})")
    print("="*80)
    
    # Create a visualization of results
    plt.figure(figsize=(12, 6))
    
    # Plot a horizontal bar for each sample showing correct/incorrect predictions
    sample_nums = [r['Sample'] for r in results]
    colors = ['green' if r['True Category'] == r['Predicted Category'] else 'red' for r in results]
    
    plt.barh(sample_nums, [1] * len(results), color=colors, alpha=0.6)
    
    # Add labels
    for i, result in enumerate(results):
        plt.text(0.5, result['Sample'], 
                f"{result['True Category']} → {result['Predicted Category']} ({result['Confidence']}%)", 
                ha='center', va='center', color='black')
    
    plt.yticks(sample_nums)
    plt.title('Test Results: Green = Correct, Red = Incorrect')
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    students_data = load_data(file_path)
    X, y = prepare_data(students_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Class imbalance visual
    class_counts = Counter(y_train)
    print("\nClass Distribution in Training Set:")
    for label, count in class_counts.items():
        print(f"{label}: {count} samples")

    plt.figure(figsize=(8, 5))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.xlabel("BMI Category")
    plt.ylabel("Number of Samples")
    plt.title("Class Distribution in Training Data")
    plt.tight_layout()
    plt.show()

    model, encoder, intercept_value, coef_value = train_model(X_train, y_train, X_test, y_test)
    print(f"Accuracy of each training iteration is : {acc_record}")
    
    # Display test results (show first 10 samples)
    display_test_results(model, encoder, X_test, y_test, num_samples=10)

    while True:
        print("\nSelect X-axis variable for plotting (Age will be on Y-axis):")
        print("1. BMI")
        print("2. Height")
        print("3. Weight")
        print("4. Daily Calories")
        print("5. Sleep Hours")
        print("6. Exercise Minutes")
        print("7. View Test Results")
        print("Q. Quit plotting")

        choice = input("Enter your choice: ").strip().lower()

        if choice == 'q':
            print("Exiting plotting mode.")
            break
        elif choice == '7':
            num_samples = input("How many test samples to display (press Enter for all): ")
            try:
                num_samples = int(num_samples) if num_samples.strip() else None
                display_test_results(model, encoder, X_test, y_test, num_samples=num_samples)
            except ValueError:
                print("Invalid number. Displaying all samples.")
                display_test_results(model, encoder, X_test, y_test)
            continue

        x_variable_map = {
            '1': 'bmi',
            '2': 'height',
            '3': 'weight',
            '4': 'daily_calories',
            '5': 'sleep_hours',
            '6': 'exercise_minutes'
        }

        x_variable = x_variable_map.get(choice)
        if x_variable:
            plot_predictions(model, encoder, X_test, y_test, x_variable, intercept_value, coef_value)
        else:
            print("Invalid choice. Please try again.")

    # Example prediction
    print("\nMaking an example prediction:")
    example_prediction = predict_bmi_category(model, encoder, weight=67, height=168, age=12.67,daily_calories=3140, sleep_hours=6.4,exercise_minutes=33, gender='ช.')
    print("Example Prediction:", example_prediction)