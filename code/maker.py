import numpy as np
import json, os, random

r_ai = -0.75
r_sw = 0.84
r_sl_bmi = -0.45 

def _standardize(x):
    x = np.asarray(x)
    return (x - x.mean()) / (x.std(ddof=0) + 1e-9)

def generate_pseudo_json(
    n_samples=500,
    seed=42,
    save_dir="output",
    filename="pseudo_data.json"
):
    if seed is not None:
        np.random.seed(seed); random.seed(seed)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    data = []

    # --- 1) Base independent vars: age, height, weight, sex ---
    age = np.clip(np.rint(np.random.normal(16.5, 1.0, n_samples)), 13, 19).astype(int)
    height = np.clip(np.rint(np.random.normal(162.3, 8.0, n_samples)), 145, 190).astype(int)
    weight = np.clip(np.round(np.random.normal(54.74, 14.37, n_samples), 1), 35, 110)

    sex = np.random.choice(["ชาย", "หญิง"], size=n_samples, p=[0.5, 0.5])

    # BMI
    bmi = weight / ((height/100.0) ** 2)

    Zb = _standardize(bmi)
    Za = _standardize(age)

    # --- 2) Active intensity (1..5), target corr ≈ -0.6 with BMI ---
    # latent normal with Corr(latent, BMI) ~= -0.6
    
    eps_ai = np.random.normal(0, 1, n_samples)
    ai_latent = r_ai * Zb + np.sqrt(max(1 - r_ai**2, 1e-6)) * eps_ai
    # center around ~3, scale a bit, then round & clamp to 1..5
    ai_cont = 3 + 0.9 * ai_latent
    active_intensity = np.clip(np.rint(ai_cont), 1, 5).astype(int)

    # --- 3) sumw (0..30), target corr ≈ +0.6 with BMI ---
    
    eps_sw = np.random.normal(0, 1, n_samples)
    sw_latent = r_sw * Zb + np.sqrt(max(1 - r_sw**2, 1e-6)) * eps_sw
    # map to 0..30-ish then clamp & int
    sw_cont = 15 + 6.5 * sw_latent  # center near middle, spread to use range
    sumw = np.clip(np.rint(sw_cont), 0, 30).astype(int)

    # --- 4) Sleep per week via your formula:
    #     sleep = 56 - (day - 7) * (hours - 8)
    #     Target corr with BMI ≈ -0.5; keep a tiny negative age effect.
     # main target
    eps_sl = np.random.normal(0, 1, n_samples)
    # combine BMI and small age effect (beta=-0.15), keep total variance 1
    beta_age = -0.15
    # make combined standardized driver
    # first orthogonalize Za to Zb to avoid inflating correlation with BMI
    Za_res = Za - (Za @ Zb) / (Zb @ Zb + 1e-9) * Zb
    Za_res = Za_res / (Za_res.std(ddof=0) + 1e-9)
    # allocate variance: r_sl_bmi^2 for BMI, beta_age^2 for age, rest for noise
    var_left = max(1 - (r_sl_bmi**2 + beta_age**2), 1e-6)
    sl_driver = r_sl_bmi * Zb + beta_age * Za_res + np.sqrt(var_left) * eps_sl

    # map driver to a *target* sleep (continuous), then back-solve hours/day
    # aim overall mean≈44 and sd≈8 (close to your stats) before the formula quantization
    sleep_target = 44 + 8 * sl_driver
    # keep within feasible band
    sleep_target = np.clip(sleep_target, 20, 80)

    # Choose day in 1..6 (avoid 7 to prevent division by zero in formula)
    day = np.random.randint(1, 7, size=n_samples)

    # Back-solve hours (continuous), then round to integer and recompute exact sleep
    # hours = 8 + (56 - sleep) / (day - 7)
    hours_cont = 8 + (56 - sleep_target) / (day - 7)
    # keep reasonable hours per day
    hours = np.clip(np.rint(hours_cont), 4, 12).astype(int)

    # Recompute sleep (for correlation truth) – not included in JSON output
    sleep_recomp = 56 - (day - 7) * (hours - 8)

    # --- 5) Build records in your exact JSON structure ---
    for i in range(n_samples):
        data.append({
            "age": int(age[i]),
            "sex": str(sex[i]),
            "height": int(height[i]),
            "weight": float(weight[i]),
            "active intensity": int(active_intensity[i]),
            "day": int(day[i]),
            "hours": int(hours[i]),
            "sumw": int(sumw[i])
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    try:
        import pandas as pd
        df = pd.DataFrame({
            "BMI": bmi,
            "AI": active_intensity,
            "SUMW": sumw,
            "SLEEP": sleep_recomp
        })
        corr = df.corr(numeric_only=True).round(3)
        print("Correlations (approx):\n", corr[["BMI"]])
    except Exception:
        pass

    print(f"Saved to: {path}")
    return data

generate_pseudo_json(n_samples=200, seed=73, save_dir=r"C:\Users\Admin\Documents\VScode\math project\bmi\data")
