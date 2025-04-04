import numpy as np

def predict_category_probabilities(age, height, weight, daily_calories, sleep_hours, exercise_minutes):
    def softmax(logits):
        exp_logits = np.exp(logits - np.max(logits))  # For numerical stability
        return exp_logits / np.sum(exp_logits)
    
    bmi = round(weight / ((height/100) ** 2), 2)

    # Model intercepts and coefficients
    intercepts = np.array([-0.14882, 0.02100, 0.05910, 0.06872])
    coefficients = np.array([
        [-6.07637,  1.70044,  1.29810, -2.06361, -0.00108, -0.35803, -0.02958],
        [-0.49721,  1.84711, -0.09388,  0.27586,  0.00025, -0.03498, -0.01330],
        [ 2.40752, -0.74141, -0.46572,  0.74756,  0.00041,  0.04142,  0.01110],
        [ 4.16607, -2.80613, -0.73850,  1.04019,  0.00042,  0.35159,  0.03177]
    ])
    
    # Input feature vector
    X = np.array([bmi, age, height, weight, daily_calories, sleep_hours, exercise_minutes])
    
    # Compute logits for each class
    logits = intercepts + np.dot(coefficients, X)
    
    # Convert logits to probabilities using softmax
    probabilities = softmax(logits)
    
    # Class labels
    categories = ['Underweight', 'Normal', 'Overweight', 'Obese']
    
    # Return probabilities as a dictionary
    return {categories[i]: round(probabilities[i], 5) for i in range(4)}

age = 17
height = 167 
weight = 52
daily_calories = 3000
sleep_hours = 5
exercise_minutes = 10

 
print(predict_category_probabilities(age, height, weight, daily_calories, sleep_hours, exercise_minutes))
