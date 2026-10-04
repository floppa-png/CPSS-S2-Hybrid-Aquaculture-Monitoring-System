"""
agent.py — Full-Loop Aquaculture Agent
=======================================
Pipeline:
  1. Receive 4 sensor inputs: Temp, pH, Turbidity, TDS
  2. Run XGBoost → predict DO
  3. Run PySR formulas → predict DO (for error comparison)
  4. Classify all 5 parameters (Temp, pH, DO, Turbidity, TDS) into SAFE/WARNING/UNSAFE
  5. Determine corrective actions (closed-loop health scoring)
  6. Call LLM for narrative explanation
  7. Return full response dict
"""

import joblib
import pandas as pd
import numpy as np
import copy
import requests
import time
import math

# ──────────────────────────────────────────────────────────────────────────────
# LOAD XGBoost MODEL (DO predictor: 4 inputs → DO)
# ──────────────────────────────────────────────────────────────────────────────

MODEL_PATH = r"C:\Users\vmj07\OneDrive\Desktop\CPSS\model\aquaculture_xgb_do.pkl"

try:
    xgb_model = joblib.load(MODEL_PATH)
    print("XGBoost DO model loaded.")
except Exception as e:
    xgb_model = None
    print(f"WARNING: XGBoost model not found: {e}. DO will use formula only.")

# ──────────────────────────────────────────────────────────────────────────────
# SAFE RANGES  (based on new dataset statistics + aquaculture domain knowledge)
# ──────────────────────────────────────────────────────────────────────────────

SAFE_RANGES = {
    "Temp":      (22.0, 32.0),    # °C
    "pH":        (6.5,   8.5),
    "DO":        (5.0,   12.0),   # mg/L
    "Turbidity": (10.0,  40.0),   # NTU
    "TDS":       (200.0, 800.0),  # ppm
}

WARNING_RANGES = {
    # (outer_low, outer_high) — beyond these = UNSAFE
    "Temp":      (18.0,  38.0),
    "pH":        (5.5,    9.5),
    "DO":        (3.0,   18.0),
    "Turbidity": (5.0,   65.0),
    "TDS":       (100.0, 1200.0),
}

# ──────────────────────────────────────────────────────────────────────────────
# PER-FISH SAFE & WARNING RANGES
# ──────────────────────────────────────────────────────────────────────────────

FISH_PROFILES = {
    "Tilapia": {
        "SAFE":    {"Temp": (25.0, 30.0), "pH": (7.0, 8.5), "DO": (4.0, 12.0), "Turbidity": (20.0,  50.0), "TDS": (200.0,  800.0)},
        "WARNING": {"Temp": (20.0, 35.0), "pH": (6.0, 9.0), "DO": (3.0, 15.0), "Turbidity": (10.0,  80.0), "TDS": (100.0, 1000.0)},
    },
    "Catfish": {
        "SAFE":    {"Temp": (24.0, 30.0), "pH": (6.5, 8.5), "DO": (3.0, 10.0), "Turbidity": (30.0,  80.0), "TDS": (150.0,  600.0)},
        "WARNING": {"Temp": (20.0, 34.0), "pH": (6.0, 9.0), "DO": (2.0, 14.0), "Turbidity": (15.0, 120.0), "TDS": (100.0,  900.0)},
    },
    "Salmon": {
        "SAFE":    {"Temp": (12.0, 16.0), "pH": (6.5, 8.0), "DO": (7.0, 14.0), "Turbidity": (0.0,  25.0), "TDS": (100.0,  400.0)},
        "WARNING": {"Temp":  (8.0, 18.0), "pH": (6.0, 8.5), "DO": (5.0, 16.0), "Turbidity": (0.0,  50.0), "TDS":  (50.0,  600.0)},
    },
    "Shrimp": {
        "SAFE":    {"Temp": (23.0, 30.0), "pH": (7.5, 8.5), "DO": (5.0, 10.0), "Turbidity": (20.0,  40.0), "TDS": (500.0, 1500.0)},
        "WARNING": {"Temp": (20.0, 32.0), "pH": (7.0, 9.0), "DO": (4.0, 12.0), "Turbidity": (10.0,  60.0), "TDS": (300.0, 2000.0)},
    },
    "Carp": {
        "SAFE":    {"Temp": (20.0, 28.0), "pH": (6.5, 8.5), "DO": (4.0, 12.0), "Turbidity": (20.0,  60.0), "TDS": (200.0, 1000.0)},
        "WARNING": {"Temp": (15.0, 32.0), "pH": (6.0, 9.0), "DO": (3.0, 15.0), "Turbidity": (10.0, 100.0), "TDS": (100.0, 1500.0)},
    },
}

UNITS = {
    "Temp":      "°C",
    "pH":        "",
    "DO":        "mg/L",
    "Turbidity": "NTU",
    "TDS":       "ppm",
}

# ──────────────────────────────────────────────────────────────────────────────
# PYSR SYMBOLIC FORMULAS (from formulas.txt)
# Each formula uses only the 4 sensor inputs: Temp, pH, Turbidity, TDS
# ──────────────────────────────────────────────────────────────────────────────


def predict_formula_do(Temp, pH, Turbidity, TDS, do_lag1=None):
    if do_lag1 is None:
        """DO = (((ph * (temp * 0.094596535)) + 56.216125) + ((((temp * (90.745476 - temp)) - sqrt(turbidity)) / -46.414043) + ph)) - (log(ph) * 17.393122)"""
        return (((pH * (Temp * 0.094596535)) + 56.216125) + ((((Temp * (90.745476 - Temp)) - math.sqrt(Turbidity)) / -46.414043) + pH)) - (math.log(pH) * 17.393122)
    else:
        """DO = (sqrt(temp) + sqrt(temp)) + ((do_lag1 + (ph + -25.797922)) + ((((do_lag1 * -0.11508278) * (do_lag1 * ph)) + 371.1783) / (temp - -3.4584827)))"""
        return (math.sqrt(Temp) + math.sqrt(Temp)) + ((do_lag1 + (pH + -25.797922)) + ((((do_lag1 * -0.11508278) * (do_lag1 * pH)) + 371.1783) / (Temp - -3.4584827)))


def predict_formula_temp(Temp, pH, Turbidity, TDS, temp_lag1=None):
    if temp_lag1 is None:
        try:
            return (((Turbidity + -44.441353) * ((pH - Turbidity) / (pH + ((TDS / -3.2240689) + -56.71152)))) + 34.46617) - pH
        except (ValueError, ZeroDivisionError):
            return Temp
    else:
        try:
            return ((temp_lag1 * -0.21818349) + (pH * math.log(math.log(math.log(temp_lag1 - math.log(Turbidity)))))) * ((-22.217386 / math.log(temp_lag1)) - -1.4772917)
        except (ValueError, ZeroDivisionError):
            return Temp


def predict_formula_ph(Temp, pH, Turbidity, TDS, ph_lag1=None):
    if ph_lag1 is None:
        try:
            return ((((Turbidity * -0.042180385) + (TDS + ((-0.26137692 / math.sqrt(Turbidity)) * (Temp + 12.055943)))) * 0.99963117) + 9.974293) - TDS
        except (ValueError, ZeroDivisionError):
            return pH
    else:
        try:
            return (((((ph_lag1 * (Turbidity / Temp)) + Temp) * -0.047429368) - -11.380294) - (4.9250216 / Turbidity)) + (-18.978714 / ph_lag1)
        except (ValueError, ZeroDivisionError):
            return pH


def predict_formula_turbidity(Temp, pH, Turbidity, TDS, turbidity_lag1=None):
    if turbidity_lag1 is None:
        try:
            return (((TDS * 0.00616665) + ((0.6343792 / math.log(math.log(pH / 1.5489146))) + -17.725893)) + (316.6758 / (Temp - 5.10029))) + Temp
        except (ValueError, ZeroDivisionError):
            return Turbidity
    else:
        try:
            return (TDS / 200.66537) + ((226.4727 / math.log(Temp)) + (((turbidity_lag1 * (((turbidity_lag1 + -37.931164) / 200.71663) / 0.7882156)) + Temp) + -68.30173))
        except (ValueError, ZeroDivisionError):
            return Turbidity


def predict_formula_tds(Temp, pH, Turbidity, TDS, tds_lag1=None):
    if tds_lag1 is None:
        return (((Temp * (Temp - 49.561565)) - -1104.1257) + Turbidity) - Temp
    else:
        return ((Turbidity + 280.18875) - (tds_lag1 * -0.28638056)) - (Temp * -1.6009117)

# ──────────────────────────────────────────────────────────────────────────────
# PYSR FORMULA RMSEs (for display)
# ──────────────────────────────────────────────────────────────────────────────


FORMULA_RMSE = {
    "DO":        4.702,
    "Temp":      5.760,
    "pH":        1.069,
    "Turbidity": 11.469,
    "TDS":       413.47,
}

# ──────────────────────────────────────────────────────────────────────────────
# XGBoost PREDICTION (DO only)
# ──────────────────────────────────────────────────────────────────────────────


def predict_xgb_do(Temp, pH, Turbidity, TDS, do_lag1=None):
    if xgb_model is None:
        return predict_formula_do(Temp, pH, Turbidity, TDS, do_lag1)
    X = pd.DataFrame([[Temp, pH, Turbidity, TDS]],
                     columns=["Temp", "pH", "Turbidity", "TDS"])
    return float(xgb_model.predict(X)[0])

# ──────────────────────────────────────────────────────────────────────────────
# RUN ALL PREDICTIONS
# ──────────────────────────────────────────────────────────────────────────────


def run_predictions(inputs):
    T = inputs["Temp"]
    ph = inputs["pH"]
    Tu = inputs["Turbidity"]
    Td = inputs["TDS"]
    do_lag1 = inputs.get("do_lag1")
    temp_lag1 = inputs.get("temp_lag1")
    ph_lag1 = inputs.get("ph_lag1")
    turbidity_lag1 = inputs.get("turbidity_lag1")
    tds_lag1 = inputs.get("tds_lag1")

    # XGBoost predicts DO only
    xgb_do = predict_xgb_do(T, ph, Tu, Td, do_lag1)

    # For Temp, pH, Turbidity, TDS → sensor reading IS the "ML value"
    # (XGBoost smoothed via formula proxy — formula acts as the model here)
    ml_preds = {
        "DO":        round(xgb_do, 3),
        "Temp":      round(T, 3),
        "pH":        round(ph, 3),
        "Turbidity": round(Tu, 3),
        "TDS":       round(Td, 3),
    }

    formula_preds = {
        "DO":        round(predict_formula_do(T, ph, Tu, Td, do_lag1), 3),
        "Temp":      round(predict_formula_temp(T, ph, Tu, Td, temp_lag1), 3),
        "pH":        round(predict_formula_ph(T, ph, Tu, Td, ph_lag1), 3),
        "Turbidity": round(predict_formula_turbidity(T, ph, Tu, Td, turbidity_lag1), 3),
        "TDS":       round(predict_formula_tds(T, ph, Tu, Td, tds_lag1), 3),
    }

    errors = {
        param: round(abs(ml_preds[param] - formula_preds[param]), 4)
        for param in ml_preds
    }

    return ml_preds, formula_preds, errors

# ──────────────────────────────────────────────────────────────────────────────
# PARAMETER CLASSIFICATION
# ──────────────────────────────────────────────────────────────────────────────


def classify_value(param, value, fish_type=None):
    if fish_type and fish_type in FISH_PROFILES:
        s_lo, s_hi = FISH_PROFILES[fish_type]["SAFE"][param]
        w_lo, w_hi = FISH_PROFILES[fish_type]["WARNING"][param]
    else:
        s_lo, s_hi = SAFE_RANGES[param]
        w_lo, w_hi = WARNING_RANGES[param]

    if value < w_lo or value > w_hi:
        return "UNSAFE"
    if value < s_lo or value > s_hi:
        return "WARNING"
    return "SAFE"

# ──────────────────────────────────────────────────────────────────────────────
# OVERALL STATUS
# ──────────────────────────────────────────────────────────────────────────────


def overall_status(details):
    unsafe = sum(1 for d in details if d["level"] == "UNSAFE")
    warning = sum(1 for d in details if d["level"] == "WARNING")
    if unsafe >= 1 or warning >= 3:
        return "UNSAFE"
    if warning >= 1:
        return "WARNING"
    return "SAFE"

# ──────────────────────────────────────────────────────────────────────────────
# HEALTH SCORE
# ──────────────────────────────────────────────────────────────────────────────


def health_score(details):
    score = 100
    for d in details:
        if d["level"] == "WARNING":
            score -= 10
        elif d["level"] == "UNSAFE":
            score -= 25
    return max(0, score)

# ──────────────────────────────────────────────────────────────────────────────
# ACTION GUIDANCE
# ──────────────────────────────────────────────────────────────────────────────


PARAM_ACTION_GUIDANCE = {
    "Temp": {
        "WARNING": "Apply shading or increase water circulation to stabilize temperature.",
        "UNSAFE":  "Immediately perform partial water exchange to correct temperature.",
    },
    "pH": {
        "WARNING": "Monitor pH closely and prepare buffering agents as needed.",
        "UNSAFE":  "Gradually adjust pH using lime or freshwater dilution.",
    },
    "DO": {
        "WARNING": "Increase aeration intensity during low oxygen periods.",
        "UNSAFE":  "Activate emergency aeration immediately.",
    },
    "Turbidity": {
        "WARNING": "Reduce pond disturbance and improve mechanical filtration.",
        "UNSAFE":  "Perform sediment removal and partial water replacement.",
    },
    "TDS": {
        "WARNING": "Monitor dissolved solids and review feed and chemical inputs.",
        "UNSAFE":  "Perform large water exchange to dilute excess dissolved solids.",
    },
}

ACTION_EFFECTS = {
    "Immediately perform partial water exchange to correct temperature.": {"Temp": -4.0},
    "Activate emergency aeration immediately.":                           {"DO": +2.0},
    "Perform sediment removal and partial water replacement.":            {"Turbidity": -12.0},
    "Perform large water exchange to dilute excess dissolved solids.":    {"TDS": -150.0},
    "Gradually adjust pH using lime or freshwater dilution.":            {"pH": +0.5},
}


def evaluate_action_impact(inputs, action, fish_type=None):
    """Score improvement from taking this action."""
    before_details = _build_details(inputs, fish_type)
    before_score = health_score(before_details)

    new_state = copy.deepcopy(inputs)
    if action in ACTION_EFFECTS:
        for k, v in ACTION_EFFECTS[action].items():
            new_state[k] = new_state.get(k, 0) + v

    after_details = _build_details(new_state, fish_type)
    after_score = health_score(after_details)
    return after_score - before_score


def _build_details(inputs, fish_type=None):
    """Build detail list from a dict of {Temp, pH, DO, Turbidity, TDS}."""
    details = []
    for param in ["Temp", "pH", "DO", "Turbidity", "TDS"]:
        val = inputs.get(param, 0)
        level = classify_value(param, val, fish_type)
        details.append({"parameter": param, "level": level, "value": val})
    return details

# ──────────────────────────────────────────────────────────────────────────────
# LLM EXPLANATION
# ──────────────────────────────────────────────────────────────────────────────


def generate_explanation(status, details, fish_type=None):
    fish_line = f"Fish Species: {fish_type}\n" if fish_type else ""
    param_lines = ""
    for d in details:
        p = d["parameter"]
        val = d["ml_value"]
        unit = UNITS.get(p, "")
        level = d["level"]
        param_lines += f"  {p}: {val}{unit} — {level}\n"

    prompt = f"""You are an expert aquaculture pond manager. Be direct and action-oriented.

{fish_line}Overall Status: {status}
Current Readings:
{param_lines}
In 1-2 sentences, state what these readings mean for the fish right now.
Then list only the actions needed to fix any WARNING or UNSAFE parameters, each starting with "* ".
If everything is SAFE, say so in one sentence — no actions needed.
Use plain text only. No markdown, no bold, no HTML."""

    try:
        start = time.time()
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model":  "llama3.1:8b-instruct-q4_K_M",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_ctx": 2048},
            },
            timeout=30,
        )
        latency = round(time.time() - start, 3)
        print(f"LLM latency: {latency}s")
        return resp.json()["response"].strip()
    except Exception as e:
        print("LLM error:", e)
        return "Local AI explanation unavailable. Check Ollama service."

# ──────────────────────────────────────────────────────────────────────────────
# MAIN AGENT ASSESSMENT
# ──────────────────────────────────────────────────────────────────────────────


def agent_assessment(inputs, fish_type=None):
    """
    inputs: dict with keys Temp, pH, Turbidity, TDS
    Returns full assessment dict.
    """
    ml_preds, formula_preds, errors = run_predictions(inputs)

    # Classify ML-predicted values (DO from XGBoost; others = sensor readings)
    details = []
    actions = []
    for param in ["Temp", "pH", "DO", "Turbidity", "TDS"]:
        val = ml_preds[param]
        level = classify_value(param, val, fish_type)

        # Formula classification
        f_val = formula_preds[param]
        f_level = classify_value(param, f_val, fish_type)

        details.append({
            "parameter":       param,
            "level":           level,
            "ml_value":        ml_preds[param],
            "formula_value":   formula_preds[param],
            "error":           errors[param],
            "formula_level":   f_level,
            "unit":            UNITS[param],
        })

        if level in ("WARNING", "UNSAFE"):
            actions.append(PARAM_ACTION_GUIDANCE[param][level])

    status = overall_status(details)
    score = health_score(details)

    # Closed-loop action selection
    final_actions = []
    if status != "SAFE":
        candidates = list(set(actions))
        scored_actions = []
        for a in candidates:
            if a in ACTION_EFFECTS:
                # Build a combined state for scoring
                state_for_eval = {
                    "Temp":      ml_preds["Temp"],
                    "pH":        ml_preds["pH"],
                    "DO":        ml_preds["DO"],
                    "Turbidity": ml_preds["Turbidity"],
                    "TDS":       ml_preds["TDS"],
                }
                delta = evaluate_action_impact(state_for_eval, a, fish_type)
                scored_actions.append((delta, a))
        if scored_actions:
            scored_actions.sort(reverse=True)
            final_actions = [a for _, a in scored_actions if _ > 0]

    llm_text = generate_explanation(status, details, fish_type)

    return {
        "status":              status,
        "health_score":        score,
        "ml_predictions":      ml_preds,
        "formula_predictions": formula_preds,
        "errors":              errors,
        "details":             details,
        "general_advice":      llm_text,
        "parameter_actions":   final_actions,
    }

# ──────────────────────────────────────────────────────────────────────────────
# TOMORROW SIMULATION
# ──────────────────────────────────────────────────────────────────────────────


def simulate_tomorrow(inputs):
    """Simple drift model for t+24h projection."""
    future = copy.deepcopy(inputs)
    future["Temp"] = round(inputs["Temp"] + 0.3 * (28.0 - inputs["Temp"]), 3)
    future["pH"] = round(inputs["pH"] - 0.05, 3)
    future["Turbidity"] = round(inputs["Turbidity"] * 1.05, 3)
    future["TDS"] = round(inputs["TDS"] * 1.02, 3)
    return future
