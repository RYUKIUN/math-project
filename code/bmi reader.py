import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import math

# File path to the json data
file_path = r"C:\Users\Admin\Documents\VScode\math project\bmi\data\updated_data.json"

def bmr_and_tdee(weight,high,age,sex,workout):
    match workout.lower():
        case "no" : factor = 1.2
        case "light" : factor = 1.375
        case "normal" : factor = 1.55
        case "active" : factor = 1.725
        case "high" : factor = 1.9
        case "" : print("workout value error")
    
    if sex == "ช.":
        bmr = 88.362 + (13.397 * weight) + (4.799 * high) - (5.677 * age)    
    if sex == "ญ.":
        bmr = 447.593 + (9.247 * weight) + (3.098 * high) - (4.33 * age)
    return bmr


def bri(wrist,heigh):
    if wrist/heigh > 1:
        print("value error : BRI ratio must be less or equal to 1")
    bri = 364.2 - (365.5*(math.sqrt(1-((wrist/heigh)**2))))
    return bri

def age_cap(age):
    if age > 18:
        age = 18
    if age < 12:
        age = 12
    return age
    
# Load JSON data
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Function to calculate BMI
def calculate_bmi(weight, height):
    height_m = height / 100
    return round(weight / (height_m ** 2), 2)

# Function to classify BMI
def classify_bmi(bmi, age, gender):
    # Define BMI thresholds for boys and girls
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

    # Get the thresholds for the specified gender and age
    overweight_threshold, obesity_threshold = bmi_thresholds[gender][age]

    # Classify BMI
    if bmi < 18.5:
        return "Underweight", (bmi / 35) * 100
    elif bmi < overweight_threshold:
        return "Normal weight", (bmi / 35) * 100
    elif bmi < obesity_threshold:
        return "Overweight", (bmi / 35) * 100
    else:
        return "Obese", (bmi / 35) * 100

# Prepare dataset with additional features
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
            gender = student.get("เพศ", "").lower()  # Ensure gender is lowercase
            
            bmi = calculate_bmi(weight, height)
            label, bmi_percentage = classify_bmi(bmi, age, gender)
            
            # Add new features to X
            X.append([bmi, age, bmi_percentage, daily_calories, sleep_hours, exercise_minutes])
            y.append(label)
        except (ValueError, KeyError) as e:
            print(f"Error processing student data: {e}")
    return np.array(X), np.array(y)

# Train and optimize model with training progress
def train_model(X_train, y_train, X_test, y_test):
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    best_model = None
    best_accuracy = 0
    for max_iter in [1000, 2000, 3000]:
        print(f"\nTraining Logistic Regression Model with max_iter={max_iter}...\n")
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=max_iter, verbose=0)  # Enable progress
        model.fit(X_train, y_train_encoded)
        
        acc = accuracy_score(y_test_encoded, model.predict(X_test))
        print(f"Accuracy after {max_iter} iterations: {acc:.2%}")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model

        if acc >= 0.85:
            break
    
    np.set_printoptions(formatter={'float': '{: .5f}'.format})
    print(f"\nFinal Model Accuracy: {best_accuracy:.2%}")
    print("Intercepts:", best_model.intercept_)
    print("Coefficients:", best_model.coef_)
    return best_model, encoder

# Function to predict BMI category
def predict_bmi_category(model, encoder, weight, height, age, daily_calories, sleep_hours, exercise_minutes, gender):
    bmi = calculate_bmi(weight, height)
    input_data = np.array([[bmi, age, (bmi / 25) * 100, daily_calories, sleep_hours, exercise_minutes]])
    predicted_label_index = model.predict(input_data)[0]
    predicted_label = encoder.inverse_transform([predicted_label_index])[0]
    probabilities = model.predict_proba(input_data)[0]
    confidence = round(np.max(probabilities) * 100, 2)
    return {"BMI": bmi, "Category": predicted_label, "Confidence (%)": confidence}

# Plot predictions with user-selected Y-axis variable
def plot_predictions(model, encoder, X_test, y_test, y_variable):
    y_pred = model.predict(X_test)
    y_pred_labels = encoder.inverse_transform(y_pred)
    colors = {'Underweight': 'blue', 'Normal weight': 'green', 'Overweight': 'orange', 'Obese': 'red'}
    
    # Map the user's choice to the corresponding column index in X_test
    y_variable_mapping = {
        'bmi_percentage': 2,
        'daily_calories': 3,
        'sleep_hours': 4,
        'exercise_minutes': 5
    }
    
    y_column_index = y_variable_mapping.get(y_variable)
    if y_column_index is None:
        print("Invalid selection for Y-axis variable.")
        return

    plt.figure(figsize=(8, 6))
    for i in range(len(X_test)):
        plt.scatter(X_test[i][1], X_test[i][y_column_index], color=colors[y_pred_labels[i]], 
                    label=y_pred_labels[i] if y_pred_labels[i] not in colors else "")
    
    
    plt.ylabel(y_variable.replace('_', ' ').title())
    plt.title("Obesity Classification")
    plt.legend(colors.keys())
    plt.show()

# Main execution
if __name__ == "__main__":
    students_data = load_data(file_path)
    X, y = prepare_data(students_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model, encoder = train_model(X_train, y_train, X_test, y_test)

    # User input for Y-axis variable
    print("Select Y-axis variable for plotting:")
    print("1. BMI Percentage")
    print("2. Daily Calories")
    print("3. Sleep Hours")
    print("4. Exercise Minutes")
    choice = input("Enter the number corresponding to your choice: ")

    y_variable_map = {
        '1': 'bmi_percentage',
        '2': 'daily_calories',
        '3': 'sleep_hours',
        '4': 'exercise_minutes'
    }

    y_variable = y_variable_map.get(choice)
    if y_variable:
        plot_predictions(model, encoder, X_test, y_test, y_variable)
    else:
        print("Invalid choice. Please run the program again.")
    
    # Example prediction with new features
    example_prediction = predict_bmi_category(model, encoder, weight=70, height=175, age=18, daily_calories=2500, sleep_hours=7, exercise_minutes=30, gender='boy')
    print("Example Prediction:", example_prediction)
