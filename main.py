from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
from fuzzywuzzy import process
import ast
import os
import json

app = Flask(__name__)
CORS(app)

# Ensure templates/static changes are reflected without restart
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


# Loading the datasets from Kaggle website

sym_des = pd.read_csv("dataset/symptoms_df.csv")
precautions = pd.read_csv("dataset/precautions_df.csv")
workout = pd.read_csv("dataset/workout_df.csv")
description = pd.read_csv("dataset/description.csv")
medications = pd.read_csv("dataset/medications.csv")
diets = pd.read_csv("dataset/diets.csv")

# Model (kept as fallback)
Rf = pickle.load(open('model/RandomForest.pkl','rb'))


# Here we make a dictionary of symptoms and diseases and preprocess it

symptoms_list = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

symptoms_list_processed = {symptom.replace('_', ' ').lower(): value for symptom, value in symptoms_list.items()}
index_to_symptom = {v: k for k, v in symptoms_list_processed.items()}

# Basic localised (Hindi / Marathi) symptom names mapped to English canonical keys.
# You can extend this dictionary over time as needed.
LOCALIZED_SYMPTOM_MAP = {
    # Hindi
    "बुखार": "high fever",
    "तेज बुखार": "high fever",
    "सर दर्द": "headache",
    "सरदर्द": "headache",
    "सिरदर्द": "headache",
    "सर्दी": "cough",
    "जुकाम": "cough",
    "नाक बहना": "runny nose",
    "खांसी": "cough",
    "उल्टी": "vomiting",
    "मतली": "nausea",
    "चक्कर": "dizziness",
    # Marathi
    "ताप": "high fever",
    "डोकेदुखी": "headache",
    "डोके दुखणे": "headache",
    "खोकला": "cough",
    "मळमळ": "nausea",
    "ओकाऱ्या": "vomiting",
}


def load_localized_symptom_map(csv_path: str = os.path.join("dataset", "localized_symptoms_map.csv")):
    """
    Optionally extend LOCALIZED_SYMPTOM_MAP from a CSV file so that
    you can maintain additional symptom synonyms without touching code.
    CSV format:
        language,localized,english_symptom
    """
    global LOCALIZED_SYMPTOM_MAP
    if not os.path.exists(csv_path):
        return
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            loc = str(row.get("localized", "")).strip().lower()
            eng = str(row.get("english_symptom", "")).strip().lower()
            if loc and eng:
                LOCALIZED_SYMPTOM_MAP[loc] = eng
    except Exception:
        # If anything goes wrong we simply fall back to the in-code defaults.
        pass


def normalise_localised_symptom(symptom: str) -> str:
    """
    Map common Hindi / Marathi symptom phrases to the English
    symptom names used by the model. Falls back to the original
    string if no mapping is found.
    """
    s = symptom.strip().lower()
    return LOCALIZED_SYMPTOM_MAP.get(s, s)


# Load additional localised mappings from dataset file if present
load_localized_symptom_map()

# ==== Active Bayesian tables built from Training.csv ====
TRAIN_PATH = "dataset/Training.csv"
_bayes_ready = False
_diseases = []
_priors = None              # shape [D]
_conditional = None         # shape [D, S]  -> P(s=1 | d)


def build_bayes_tables(alpha: float = 1.0):
    global _bayes_ready, _diseases, _priors, _conditional
    if not os.path.exists(TRAIN_PATH):
        _bayes_ready = False
        return
    df = pd.read_csv(TRAIN_PATH)
    df.columns = [c.strip().replace('_', ' ').lower() for c in df.columns]
    label_col = None
    for cand in ["prognosis", "disease", "label"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        _bayes_ready = False
        return
    symptom_cols = [c for c in df.columns if c != label_col]
    aligned_cols = [c for c in symptom_cols if c in symptoms_list_processed]
    if not aligned_cols:
        _bayes_ready = False
        return
    _diseases = sorted(df[label_col].unique().tolist())
    D = len(_diseases)
    S = len(symptoms_list_processed)
    disease_counts = df[label_col].value_counts()
    _priors = np.array([disease_counts.get(d, 0) for d in _diseases], dtype=float)
    _priors = (_priors + alpha) / (_priors.sum() + alpha * D)
    _conditional = np.full((D, S), 0.5, dtype=float)
    for s_name, s_idx in symptoms_list_processed.items():
        if s_name in aligned_cols:
            for d_i, d_name in enumerate(_diseases):
                sub = df[df[label_col] == d_name]
                pos = sub[s_name].astype(float).sum()
                n = len(sub)
                _conditional[d_i, s_idx] = (pos + alpha) / (n + 2 * alpha)
    _bayes_ready = True


def bayes_rank(positive_symptoms: list, negative_symptoms: list = None, top_k: int = 3):
    if negative_symptoms is None:
        negative_symptoms = []
    if not _bayes_ready:
        build_bayes_tables()
    if not _bayes_ready:
        return []
    log_post = np.log(_priors + 1e-12)
    for s in positive_symptoms:
        if s in symptoms_list_processed:
            idx = symptoms_list_processed[s]
            psd = _conditional[:, idx]
            log_post += np.log(psd + 1e-9)
    for s in negative_symptoms:
        if s in symptoms_list_processed:
            idx = symptoms_list_processed[s]
            psd = 1.0 - _conditional[:, idx]
            log_post += np.log(psd + 1e-9)
    post = np.exp(log_post - log_post.max())
    post = post / post.sum()
    order = np.argsort(-post)
    results = []
    for i in order[:top_k]:
        results.append({"disease": _diseases[i], "prob": float(post[i])})
    return results


def suggest_symptoms_by_variance(candidates_count: int, provided: list, consider_top: int = 5):
    if not _bayes_ready:
        build_bayes_tables()
    if not _bayes_ready:
        return []
    top = bayes_rank(provided, [], top_k=consider_top)
    if not top:
        return []
    idxs = [ _diseases.index(t['disease']) for t in top ]
    variances = []
    provided_set = set(provided)
    for s_idx in range(len(symptoms_list_processed)):
        s_name = index_to_symptom[s_idx]
        if s_name in provided_set:
            continue
        vals = _conditional[idxs, s_idx]
        variances.append((np.std(vals), s_name))
    variances.sort(reverse=True)
    return [name for _, name in variances[:candidates_count]]


def suggest_symptoms_for_primary_disease(
    primary_disease: str,
    provided: list,
    max_suggestions: int = 5,
):
    """
    Suggest additional symptoms that are *typical* for the most likely disease,
    using the curated `symptoms_df.csv` table (sym_des).

    This is used to build follow‑up questions like:
    "Do you also have mild fever, abdominal pain, ... ?"
    """
    if not primary_disease:
        return []

    provided_set = set(provided or [])

    # Find row for this disease in the curated table
    # Normalise disease names slightly (strip spaces) for matching.
    disease_clean = str(primary_disease).strip()
    df_clean = sym_des.copy()
    df_clean["Disease"] = df_clean["Disease"].astype(str).str.strip()
    row = df_clean[df_clean["Disease"] == disease_clean]
    if row.empty:
        return []

    symptom_cols = [c for c in row.columns if c.startswith("Symptom_")]
    suggestions = []
    for col in symptom_cols:
        raw = row.iloc[0][col]
        # Skip missing values (NaN/None)
        if pd.isna(raw):
            continue
        val = str(raw).strip()
        if not val or val.lower() == "nan":
            continue
        # Normalise to match how symptoms are processed elsewhere
        pretty = val.replace("_", " ").lower()
        if pretty not in provided_set and pretty not in suggestions:
            suggestions.append(pretty)
        if len(suggestions) >= max_suggestions:
            break

    return suggestions


# Helper to extract disease information

def information(predicted_dis):
    disease_desciption = description[description['Disease'] == predicted_dis]['Description']
    disease_desciption = " ".join([w for w in disease_desciption])
    disease_precautions = precautions[precautions['Disease'] == predicted_dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    disease_precautions = [col for col in disease_precautions.values]
    disease_medications = medications[medications['Disease'] == predicted_dis]['Medication']
    disease_medications = [med for med in disease_medications.values]
    disease_diet = diets[diets['Disease'] == predicted_dis]['Diet']
    disease_diet = [die for die in disease_diet.values]
    disease_workout = workout[workout['disease'] == predicted_dis] ['workout']
    return disease_desciption, disease_precautions, disease_medications, disease_diet, disease_workout


# Function to correct the spellings of the symptom (if any)

def correct_spelling(symptom):
    closest_match, score = process.extractOne(symptom, symptoms_list_processed.keys())
    if score >= 80:
        return closest_match
    else:
        return None

@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/symptom_list', methods=['GET'])
def symptom_list():
    return jsonify(sorted(list(symptoms_list_processed.keys())))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Require a reasonable number of clearly–matched symptoms
    MIN_SYMPTOMS = 4
    # Confidence thresholds
    MIN_CONF = 0.7          # below this, we always ask for more info
    HIGH_CONF = 0.9         # above this, we can answer even with < MIN_SYMPTOMS
    TOP_K = 3

    if request.method == 'POST':
        try:
            if request.is_json:
                data = request.get_json()
                symptoms = data.get('symptoms', '')
                negatives = data.get('negatives', [])
                name = data.get('name', '')
                age = data.get('age', '')
                gender = data.get('gender', '')
            else:
                symptoms = request.form.get('symptoms', '')
                negatives = request.form.getlist('negatives')
                name = request.form.get('name', '')
                age = request.form.get('age', '')
                gender = request.form.get('gender', '')
            
            if not symptoms or symptoms == "Symptoms":
                if request.is_json:
                    return jsonify({'error': 'Please provide at least a few clear symptoms for prediction.'})
                else:
                    message = "Please either write symptoms or you have written misspelled symptoms"
                    return render_template('index.html', message=message)
            
            patient_symptoms = [s.strip() for s in symptoms.split(',') if s.strip()]
            patient_symptoms = [symptom.strip("[]' ") for symptom in patient_symptoms]

            # First normalise common Hindi / Marathi symptom words into English,
            # then spell–correct and deduplicate.
            corrected_symptoms = []
            for symptom in patient_symptoms:
                normalised = normalise_localised_symptom(symptom)
                corrected = correct_spelling(normalised)
                if corrected and corrected not in corrected_symptoms:
                    corrected_symptoms.append(corrected)

            # Normalise + spell–correct negatives (symptoms the user explicitly denies)
            corrected_negatives = []
            for symptom in negatives or []:
                normalised = normalise_localised_symptom(symptom)
                corrected = correct_spelling(normalised)
                if corrected and corrected not in corrected_negatives and corrected not in corrected_symptoms:
                    corrected_negatives.append(corrected)

            # If we could not confidently match at least one symptom, stop early
            if len(corrected_symptoms) == 0:
                return jsonify({
                    'success': False,
                    'error': 'I could not recognise any of your symptoms. Please try again with common symptom words (for example: headache, fever, cough).'
                })

            top_k = bayes_rank(corrected_symptoms, corrected_negatives, top_k=TOP_K)
            if not top_k:
                i_vector = np.zeros(len(symptoms_list_processed))
                for s in corrected_symptoms:
                    i_vector[symptoms_list_processed[s]] = 1
                if hasattr(Rf, 'predict_proba'):
                    proba = Rf.predict_proba([i_vector])[0]
                    idx_prob = list(enumerate(proba))
                    idx_prob.sort(key=lambda x: x[1], reverse=True)
                    top_k = [{'disease': diseases_list[idx], 'prob': float(p)} for idx, p in idx_prob[:TOP_K]]
                else:
                    pred = Rf.predict([i_vector])[0]
                    top_k = [{'disease': diseases_list[pred], 'prob': 1.0}]

            primary_prob = top_k[0]['prob'] if top_k else 0.0
            predicted_disease = top_k[0]['disease'] if top_k else None
            # Decide whether we really need more symptoms:
            # - If confidence is very high (>= HIGH_CONF), we accept even with
            #   fewer than MIN_SYMPTOMS to avoid endless questioning.
            # - If confidence is low (< MIN_CONF), we keep asking.
            # - In the middle range, we require at least MIN_SYMPTOMS.
            if primary_prob < MIN_CONF:
                need_more = True
            elif primary_prob >= HIGH_CONF:
                need_more = False
            else:
                need_more = len(corrected_symptoms) < MIN_SYMPTOMS
            if need_more:
                # Prefer asking for symptoms that are typical for the
                # most likely disease so far (for example, if hepatitis A
                # is at 95%, we only ask about classic hepatitis symptoms).
                suggestions = suggest_symptoms_for_primary_disease(
                    predicted_disease,
                    corrected_symptoms + corrected_negatives,
                    max_suggestions=5,
                )

                # If we cannot find curated symptoms for this disease,
                # don't mix in unrelated ones; just skip suggestions.
                return jsonify({
                    'success': True,
                    'need_more': True,
                    'symptoms': corrected_symptoms,
                    'negatives': corrected_negatives,
                    'top_k': top_k,
                    'suggested_symptoms': suggestions
                })

            predicted_disease = predicted_disease or top_k[0]['disease']
            # Qualitative confidence label for the UI
            if primary_prob >= 0.85:
                confidence_label = "high"
            elif primary_prob >= 0.6:
                confidence_label = "moderate"
            else:
                confidence_label = "low"

            dis_des, precautions_vals, medications_vals, rec_diet, workout_vals = information(predicted_disease)

            my_precautions = []
            for i in precautions_vals[0]:
                my_precautions.append(i)

            medication_list = ast.literal_eval(medications_vals[0])
            medications_list = []
            for item in medication_list:
                medications_list.append(item)

            diet_list = ast.literal_eval(rec_diet[0])
            rec_diet_list = []
            for item in diet_list:
                rec_diet_list.append(item)

            workout_list = []
            if hasattr(workout_vals, 'values'):
                for item in workout_vals.values:
                    workout_list.append(item)
            else:
                workout_list = list(workout_vals) if workout_vals is not None else []

            return jsonify({
                'success': True,
                'need_more': False,
                'name': name,
                'age': age,
                'gender': gender,
                'symptoms': corrected_symptoms,
                'negatives': corrected_negatives,
                'top_k': top_k,
                'primary_confidence': float(primary_prob),
                'confidence_label': confidence_label,
                'symptom_count': len(corrected_symptoms),
                'negative_count': len(corrected_negatives),
                'predicted_disease': predicted_disease,
                'dis_des': dis_des,
                'my_precautions': my_precautions,
                'medications': medications_list,
                'my_diet': rec_diet_list,
                'workout': workout_list
            })
        
        except Exception as e:
            error_msg = f"An error occurred during prediction: {str(e)}"
            if request.is_json:
                return jsonify({'error': error_msg})
            else:
                return render_template('index.html', message=error_msg)

    return render_template('index.html')


# about view function and path
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    build_bayes_tables()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)), debug=True)                  
    