import json
import random

# Load the JSON data
file_path = r"C:\Users\Admin\Documents\VScode\math project\bmi\data\main data.json"

with open(file_path, "r", encoding="utf-8") as file:
    student_data = json.load(file)

# Function to generate random but reasonable values
def generate_additional_data(student):
    weight = float(student["น้ำหนัก"])
    height = float(student["ส่วนสูง"])  # Ensure height is treated as a float
    age = int(student["อายุ(ปี)"])

    # Estimate calories based on weight and age
    if weight < 40:
        calories = random.randint(1600, 1900)
    elif weight < 50:
        calories = random.randint(1900, 2100)
    elif weight < 60:
        calories = random.randint(2100, 2500)
    elif weight < 65:
        calories = random.randint(2500,2800)

    else:
        calories = random.randint(2700, 3200)

    # Estimate sleep hours based on age (general recommendation)
    sleep_hours = round(random.uniform(5.5, 8), 1)

    # Estimate workout minutes per day based on weight and height
    bmi = weight / ((height / 100) ** 2)
    if bmi > 25:  # Higher BMI → more workout encouraged
        workout_minutes = random.randint(20, 60)
    else:
        workout_minutes = random.randint(10, 40)

    return {
        "เเคลอรี่ต่อวัน": calories,
        "ชั่วโมงการนอน": sleep_hours,
        "เวลาออกกำลังกาย(นาที)": workout_minutes
    }

# Add the new data to each student record
for student in student_data:
    additional_data = generate_additional_data(student)
    student.update(additional_data)

# Save the updated data
updated_file_path = r"C:\Users\Admin\Documents\VScode\math project\bmi\data\updated_data.json"
with open(updated_file_path, "w", encoding="utf-8") as file:
    json.dump(student_data, file, ensure_ascii=False, indent=4)

print("Updated file saved as:", updated_file_path)
