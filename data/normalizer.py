input_path = r"C:\Users\Admin\Documents\VScode\math project\bmi\data\data.json"
output_path = r"C:\Users\Admin\Documents\VScode\math project\bmi\data\normalized_data.json"
adaptor = r"C:\Users\Admin\Documents\VScode\math project\bmi\data\adaptor.json"

import json
import numpy as np


def sleeptime_per_week(day, hour):
    return 56 - ((day - 7) * (hour - 8))

def encode_sex(sex):
    return 0 if sex == "ชาย" else 1

def classify(weight, height, age, gender):
    height_m = height / 100
    bmi = round(weight / (height_m ** 2), 2)
    bmi_thresholds = {
        'ช.': {
            12: 21.9, 13: 22.6, 14: 23.3, 15: 23.9,
            16: 24.6, 17: 25.1, 18: 25.7
        },
        'ญ.': {
            12: 21.7, 13: 22.4, 14: 23.2, 15: 23.8,
            16: 24.5, 17: 25.1, 18: 25.7
        }
    }
    gender_key = 'ช.' if gender == "ชาย" else 'ญ.'
    age = max(12, min(18, int(age)))
    threshold = bmi_thresholds[gender_key][age]
    
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < threshold:
        return "Normal"
    else:
        return "Overweight"

with open(input_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

processed = []
features_matrix = []

for entry in raw_data:
    sex_label = entry["sex"]
    sex_val = encode_sex(sex_label)
    age = float(entry["age"])
    weight = float(entry["weight"])
    height = float(entry["height"])
    active_intensity = float(entry["active intensity"])
    day = float(entry["day"])
    hours = float(entry["hours"])
    sumw = float(entry["sumw"]) 

    sleep = sleeptime_per_week(day, hours)
    bmi_class = classify(weight, height, age, sex_label)

    features_matrix.append([age, weight, height, active_intensity, sleep, sumw])

    processed.append({
        "sex": sex_val,
        "age": age,
        "weight": weight,
        "height": height,
        "active_intensity": active_intensity,
        "sleep_per_week": sleep,
        "sumw": sumw,
        "class": bmi_class
    })

features_array = np.array(features_matrix)
mean = features_array.mean(axis=0)
std = features_array.std(axis=0)

standardized_output = []
for entry in processed:
    standardized_entry = {
        "sex": entry["sex"],
        "age": round((entry["age"] - mean[0]) / std[0], 6),
        "weight": round((entry["weight"] - mean[1]) / std[1], 6),
        "height": round((entry["height"] - mean[2]) / std[2], 6),
        "active_intensity": round((entry["active_intensity"] - mean[3]) / std[3], 6),
        "sleep_per_week": round((entry["sleep_per_week"] - mean[4]) / std[4], 6),
        "sumw": round((entry["sumw"] - mean[5]) / std[5], 6),
        "class": entry["class"]
    }
    standardized_output.append(standardized_entry)


with open(output_path, "w", encoding="utf-8") as f:
    json.dump(standardized_output, f, ensure_ascii=False, indent=2)

feature_names = ["age", "weight", "height", "active_intensity", "sleep_per_week", "sumw"]

adapter = {
    "mean": {name: round(mean[i], 6) for i, name in enumerate(feature_names)},
    "std": {name: round(std[i], 6) for i, name in enumerate(feature_names)}
}

with open(adaptor, "w", encoding="utf-8") as f:
    json.dump(adapter, f, ensure_ascii=False, indent=2)


print("create - normalized_data.json") 
print("create - adapter.json")
