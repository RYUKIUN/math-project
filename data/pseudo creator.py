output = r"C:\Users\Admin\Documents\VScode\math project\new refined\data\pseudodata.json"

import json
import numpy as np
import random

np.random.seed(2)
random.seed(2)

# Fixed height per age/sex for low noise
HEIGHTS = {
    15: {"ชาย": 168, "หญิง": 157},
    16: {"ชาย": 169, "หญิง": 158},
    17: {"ชาย": 170, "หญิง": 159},
    18: {"ชาย": 171, "หญิง": 160}
}

# BMI distribution range (broad)
BMI_RANGES = [17, 19, 21, 23, 25, 27, 29, 31, 33]  # 9 values across categories

def generate_fixed_entry(age):
    sex = random.choice(["ชาย", "หญิง"])
    height = HEIGHTS[age][sex]
    height_m = height / 100
    bmi = random.choice(BMI_RANGES) + np.random.uniform(-0.3, 0.3)  # tiny jitter
    weight = round(bmi * (height_m ** 2), 1)

    multiplier = 2  # fixed for correlation
    if sex == "ชาย":
        bmr = 66 + (13.7 * weight) + (5 * height) - (6.8 * age)
    else:
        bmr = 655 + (9.6 * weight) + (1.8 * height) - (4.7 * age)

    calorie = round(bmr * multiplier, 2)
    day = random.randint(1, 7)
    hours = random.randint(2, 8)

    return {
        "age": age,
        "sex": sex,
        "height": round(height, 1),
        "weight": round(weight, 1),
        "multiplier": multiplier,
        "day": day,
        "hours": hours
    }

# Generate 50 per age (200 total)
pseudodata = []
for age in [15, 16, 17, 18]:
    pseudodata.extend([generate_fixed_entry(age) for _ in range(50)])

# Save file
with open(output, "w", encoding="utf-8") as f:
    json.dump(pseudodata, f, ensure_ascii=False, indent=2)
print("✅ Ultra-high correlation data generated.")

