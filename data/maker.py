import numpy as np
import json
import os
import random

def generate_pseudo_json(n_samples=100, seed=None, save_dir="output", filename="ps-data.json"):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)

    data_list = []

    for _ in range(n_samples):
        # --- Base independent variables ---
        age = int(np.random.normal(16.5, 1.0))  # teens
        age = max(13, min(19, age))  # clamp between 13–19
        weight = round(np.random.normal(55, 14), 1)
        height = int(np.random.normal(162, 8))
        sex = random.choice(["ชาย", "หญิง"])
        day = random.randint(1, 7)
        hours = random.randint(4, 10)  # realistic sleep per day

        # --- Derived BMI ---
        bmi = weight / ((height / 100) ** 2)

        # --- Sleep per week (negative corr with age) ---
        base_sleep = 56 - (day - 7) * (hours - 8)
        noise = np.random.normal(0, 3)
        sleep_per_week = base_sleep - 0.3 * (age - 16) + noise
        sleep_per_week = max(0, min(80, sleep_per_week))

        # --- Active intensity (1–5, negative corr with BMI) ---
        noise = np.random.normal(0, 0.8)
        active_intensity = 3.5 - 0.08 * bmi + noise
        active_intensity = int(round(active_intensity))
        active_intensity = max(1, min(5, active_intensity))

        # --- Sumw (0–30, positive corr with BMI) ---
        noise = np.random.normal(0, 3)
        sumw = 10 + 0.5 * bmi + noise
        sumw = max(0, min(30, int(sumw)))

        entry = {
            "age": age,
            "sex": sex,
            "height": height,
            "weight": weight,
            "active intensity": active_intensity,
            "day": day,
            "hours": hours,
            "sumw": sumw,
            "sleep_per_week": round(sleep_per_week, 0),
        }

        data_list.append(entry)

    # Save JSON file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

    print(f"File saved to: {file_path}")
    return data_list

# Example usage
dataset = generate_pseudo_json(n_samples=200, seed=42, save_dir=r"C:\Users\Admin\Documents\VScode\math project\bmi\data")
