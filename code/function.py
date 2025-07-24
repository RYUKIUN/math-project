def classify(weight, height, age, gender):
    # Calculate BMI
    height_m = height / 100
    bmi = round(weight / (height_m ** 2), 2)
    
    # BMI thresholds by age and gender
    bmi_thresholds = {
        'ช.': {
            12: (21.9),
            13: (22.6),
            14: (23.3),
            15: (23.9),
            16: (24.6),
            17: (25.1),
            18: (25.7)
        },
        'ญ.': {
            12: (21.7),
            13: (22.4),
            14: (23.2),
            15: (23.8),
            16: (24.5),
            17: (25.1),
            18: (25.7)
        }
    }
    
    # Cap age to available range
    age = max(12, min(18, int(age)))
    
    # Get thresholds for the specific age and gender
    overweight_threshold = bmi_thresholds[gender][age]
    
    # 2-class classification: Normal vs Not Normal
    if 18.5 <= bmi < overweight_threshold:
        classification = "Normal"
    else:
        classification = "Not Normal"
    
    return classification

def Calories(sex,age,weight,height,multiplyer):
    base = (10 * weight) + (6.25 * height) - (5 * age)
    if sex == "ชาย":
        base += 5
    elif sex == "หญิง":
        base -= 161

    table = [1.2,1.375,1.55,1.7,1.9]

    return base * table[int(multiplyer) - 1]

def sleeptime_per_week(day,hour):
    return 56 - ((day - 7) * (hour - 8))

    