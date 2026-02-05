import streamlit as st
import numpy as np
import pandas as pd
import joblib
import io
import os
import textwrap
from datetime import datetime
from PIL import Image

try:
    from tensorflow.keras.models import load_model  
    TENSORFLOW_AVAILABLE = True
except ImportError:
    load_model = None
    TENSORFLOW_AVAILABLE = False

# ---------------- Config & CSS ----------------
st.set_page_config(page_title="🏥 Neuroblastoma Risk Predictor", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #ffffff; color: #0b2a4a; margin-top: -40px !important; } /* reduces top gap */
    .card { background: #f8fafc; padding: 12px; border-radius: 8px; }
    div.stButton > button:first-child {
        width: 100%;
        background-color: #0b66c3;
        color: white;
        font-size: 18px;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.85em 0;
    }
    div.stButton > button:first-child:hover { background-color: #094c8d; }
    .risk-dot { display:inline-block; width:18px; height:18px; border-radius:50%; margin-right:8px; vertical-align:middle; }
    .footer { text-align:center; color:gray; padding:10px 0; margin-top:18px; }
    .small-muted { color:#6b7280; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Files & constants ----------------
PATIENTS_CSV = "patients.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

SCAN_MODEL_PATH = "neuro_model.keras"  

# ---------------- Translations (FIXED - COMPLETE) ----------------
translations = {
    "English": {
        "fill_patient_note": "Please fill in patient details.",
        "assessment_date": "Assessment date",
        "age": "Age (years)",
        "gender": "Gender",
        "male": "Male",
        "female": "Female",
        "other": "Other",
        "past_patient_data": "Past patient data",
        "title": "🏥 Neuroblastoma Risk Predictor",
        "disclaimer": "***This tool is for educational purposes only and does not replace professional medical advice. This app is supposed to give an idea of your child's Neuroblastoma risk. Always contact a health professional for all medical decisions.***",
        "nutshell_title": "Neuroblastoma in a Nutshell",
        "nutshell_text": "Neuroblastoma is a rare cancer that forms in immature nerve cells (neuroblasts), most commonly in infants and young children under age 5, often starting in the abdomen near the adrenal glands. It arises from the sympathetic nervous system and can spread to bones, bone marrow, lymph nodes, or liver. This app estimates neuroblastoma risk based on symptoms, lab results, and optional scan analysis.",
        "major_symptoms": "🩺 Major symptoms",
        "rarer_symptoms": "🎗️ Rarer symptoms",
        "additional_symptoms": "➕ Additional symptoms",
        "lab_results_title": "🔬 Lab and genetics",
        "scan_section_title": "🩻 Scan analysis (experimental)",
        "scan_uploader_label": "Upload a scan image (JPG/PNG)",
        "scan_model_not_available": "Imaging model not available on this server.",
        "scan_analyze_button": "🔬 Analyze scan",
        "scan_neuro_text": "Scan suggests neuroblastoma",
        "scan_non_neuro_text": "Scan does not suggest neuroblastoma",
        "scan_probability_label": "Scan neuroblastoma probability:",
        "scan_prediction_label": "Scan prediction:",
        "final_combined": "Combined scan + symptoms risk",
        "risk_low": "Low risk",
        "risk_mild": "Mild risk", 
        "risk_moderate": "Moderate risk",
        "risk_high": "High risk",
    "suggestions_low": textwrap.dedent("""\
        • Low risk
        • Routine follow-up may be enough.
        • Monitor for new symptoms."""),  
    
    "suggestions_mild": textwrap.dedent("""\     
        • Mild risk
        • Discuss with a pediatrician.
        • Consider basic screening tests."""),
    
    "suggestions_moderate": textwrap.dedent("""\
        • Moderate risk
        • Specialist evaluation recommended.
        • Imaging and lab tests advised."""),
    
    "suggestions_high": textwrap.dedent("""\
        • **High risk**
        • Urgent specialist evaluation recommended.
        • Immediate imaging and specialist consult."""),
    
    "risk_ref_title": "Risk Reference",
    "risk_ref_text": textwrap.dedent("""\
        These probabilities are estimates based on machine learning patterns and do not replace clinical judgment.
        • Low risk = 0-40%
        • Mild risk = 41-60%
        • Moderate risk = 61-80%
        • High risk = 81-100%"""),
        "predict_button": "🚀 Predict Risk",
        "download_csv": "📥 Download this assessment as CSV",
        "download_all_csv": "📥 Download all past data as CSV",
        "store_data": "💾 Store this result in patient database",
        "feedback": "Feedback",
        "submit_feedback": "Submit feedback",
        "symptom_list": {
            "lump": "Lump in belly or chest",
            "abdominal_pain": "Abdominal pain",
            "weight_loss": "Unexplained weight loss",
            "constipation": "Constipation",
            "bone_pain": "Bone pain",
            "bulging_eyes": "Bulging eyes or dark circles",
            "fever": "Fever",
            "fatigue": "Fatigue",
            "cough": "Cough",
            "runny_nose": "Runny nose",
            "sore_throat": "Sore throat",
            "aches": "Body aches",
            "unexplained_pain": "Unexplained pain",
            "high_bp": "High blood pressure",
            "vomiting": "Vomiting",
            "mycn": "MYCN amplification",
            "alk": "ALK mutation",
            "deletion_11q": "11q deletion",
            "gain_17q": "17q gain"
        }
    },
    "Spanish": {
        "fill_patient_note": "Por favor complete los detalles del paciente.",
        "assessment_date": "Fecha de evaluación",
        "age": "Edad (años)",
        "gender": "Género",
        "male": "Masculino",
        "female": "Femenino", 
        "other": "Otro",
        "past_patient_data": "Datos de pacientes anteriores",
        "title": "🏥 Predictor de Riesgo de Neuroblastoma",
        "disclaimer": "***Esta herramienta es solo para fines educativos y no sustituye la consulta médica profesional. Su objetivo es proporcionar una idea del riesgo de neuroblastoma de su hijo. Siempre consulte con un profesional de la salud para cualquier decisión médica.***",
        "nutshell_title": "En resumen",
        "nutshell_text": "El neuroblastoma es un cáncer poco común que se forma en células nerviosas inmaduras (neuroblastos), con mayor frecuencia en bebés y niños pequeños menores de 5 años, y suele comenzar en el abdomen, cerca de las glándulas suprarrenales. Se origina en el sistema nervioso simpático y puede propagarse a los huesos, la médula ósea, los ganglios linfáticos o el hígado. Esta aplicación estima el riesgo de neuroblastoma basándose en síntomas, resultados de laboratorio y una exploración opcional.",
        "major_symptoms": "Síntomas principales",
        "rarer_symptoms": "Síntomas más raros",
        "additional_symptoms": "Síntomas adicionales",
        "lab_results_title": "Laboratorio y genética",
        "scan_section_title": "Análisis de escaneo (experimental)",
        "scan_uploader_label": "Subir imagen de escaneo (JPG/PNG)",
        "scan_model_not_available": "Modelo de imagen no disponible en este servidor.",
        "scan_analyze_button": "🔬 Analizar escaneo",
        "scan_neuro_text": "El escaneo sugiere neuroblastoma",
        "scan_non_neuro_text": "El escaneo no sugiere neuroblastoma",
        "scan_probability_label": "Probabilidad de neuroblastoma del escaneo:",
        "scan_prediction_label": "Predicción del escaneo:",
        "final_combined": "Riesgo combinado de escaneo + síntomas",
        "risk_low": "Bajo riesgo",
        "risk_mild": "Riesgo leve",
        "risk_moderate": "Riesgo moderado",
        "risk_high": "Alto riesgo",
        "suggestions_low": "• Bajo riesgo; seguimiento rutinario puede ser suficiente.\n• Monitorear nuevos síntomas.",
        "suggestions_mild": "• Riesgo leve; consultar con pediatra.\n• Considerar pruebas de screening básicas.",
        "suggestions_moderate": "• Riesgo moderado; evaluación especialista recomendada.\n• Pruebas de imagen y laboratorio aconsejadas.",
        "suggestions_high": "• **Alto riesgo; evaluación especialista urgente recomendada.**\n• Imagen inmediata y consulta especialista.",
        "risk_ref_title": "Cómo interpretar el riesgo",
        "risk_ref_text": "Estas probabilidades son estimaciones basadas en patrones de aprendizaje automático y no reemplazan el juicio clínico.",
        "predict_button": "🚀 Predecir Riesgo",
        "download_csv": "📥 Descargar esta evaluación como CSV",
        "download_all_csv": "📥 Descargar todos los datos anteriores como CSV",
        "store_data": "💾 Guardar este resultado en base de datos de pacientes",
        "feedback": "Comentarios",
        "submit_feedback": "Enviar comentarios",
        "symptom_list": {
            "lump": "Bulto en abdomen o pecho",
            "abdominal_pain": "Dolor abdominal",
            "weight_loss": "Pérdida de peso inexplicable",
            "constipation": "Estreñimiento",
            "bone_pain": "Dolor óseo",
            "bulging_eyes": "Ojos saltones o círculos oscuros",
            "fever": "Fiebre",
            "fatigue": "Fatiga",
            "cough": "Tos",
            "runny_nose": "Nariz mocosa",
            "sore_throat": "Dolor de garganta",
            "aches": "Dolores corporales",
            "unexplained_pain": "Dolor inexplicable",
            "high_bp": "Presión arterial alta",
            "vomiting": "Vómitos",
            "mycn": "Amplificación MYCN",
            "alk": "Mutación ALK",
            "deletion_11q": "Deleción 11q",
            "gain_17q": "Ganancia 17q"
        }
    },
    "French": {
        "fill_patient_note": "Veuillez remplir les détails du patient.",
        "assessment_date": "Date d'évaluation",
        "age": "Âge (années)",
        "gender": "Genre",
        "male": "Masculin",
        "female": "Féminin",
        "other": "Autre",
        "past_patient_data": "Données patients passées",
        "title": "🏥 Prédicteur de Risque Neuroblastome",
        "disclaimer": "***Cet outil est destiné à des fins éducatives uniquement et ne remplace pas un avis médical professionnel. Cette application vise à vous donner une idée du risque de neuroblastome chez votre enfant. Consultez toujours un professionnel de la santé pour toute décision médicale.***",
        "nutshell_title": "En résumé",
        "nutshell_text": "Le neuroblastome est un cancer rare qui se développe à partir de cellules nerveuses immatures (neuroblastes), le plus souvent chez les nourrissons et les jeunes enfants de moins de 5 ans, généralement dans l'abdomen, près des glandes surrénales. Issu du système nerveux sympathique, il peut se propager aux os, à la moelle osseuse, aux ganglions lymphatiques ou au foie. Cette application évalue le risque de neuroblastome en fonction des symptômes, des résultats d'analyses de laboratoire et, en option, d'une analyse d'imagerie.",
        "major_symptoms": "Symptômes principaux",
        "rarer_symptoms": "Symptômes plus rares",
        "additional_symptoms": "Symptômes additionnels",
        "lab_results_title": "Laboratoire et génétique",
        "scan_section_title": "Analyse scanner (expérimental)",
        "scan_uploader_label": "Télécharger image scanner (JPG/PNG)",
        "scan_model_not_available": "Modèle d'imagerie non disponible sur ce serveur.",
        "scan_analyze_button": "🔬 Analyser scanner",
        "scan_neuro_text": "Scanner suggère neuroblastome",
        "scan_non_neuro_text": "Scanner ne suggère pas neuroblastome",
        "scan_probability_label": "Probabilité neuroblastome scanner:",
        "scan_prediction_label": "Prédiction scanner:",
        "final_combined": "Échelle de risque combinée + symptômes",
        "risk_low": "Faible risque",
        "risk_mild": "Risque léger",
        "risk_moderate": "Risque modéré",
        "risk_high": "Risque élevé",
        "suggestions_low": "• Faible risque; suivi de routine peut suffire.\n• Surveiller nouveaux symptômes.",
        "suggestions_mild": "• Risque léger; discuter avec pédiatre.\n• Considérer tests de dépistage basiques.",
        "suggestions_moderate": "• Risque modéré; évaluation spécialiste recommandée.\n• Imagerie et tests labo conseillés.",
        "suggestions_high": "• **Risque élevé; évaluation spécialiste urgente recommandée.**\n• Imagerie immédiate et consultation spécialiste.",
        "risk_ref_title": "Comment interpréter le risque",
        "risk_ref_text": "Ces probabilités sont des estimations basées sur des modèles d'apprentissage automatique et ne remplacent pas le jugement clinique.",
        "predict_button": "🚀 Prédire Risque",
        "download_csv": "📥 Télécharger cette évaluation en CSV",
        "download_all_csv": "📥 Télécharger toutes les données passées en CSV",
        "store_data": "💾 Stocker ce résultat dans base patients",
        "feedback": "Retour",
        "submit_feedback": "Envoyer retour",
        "symptom_list": {
            "lump": "Masse abdominale ou thoracique",
            "abdominal_pain": "Douleur abdominale",
            "weight_loss": "Perte de poids inexpliquée",
            "constipation": "Constipation",
            "bone_pain": "Douleur osseuse",
            "bulging_eyes": "Yeux bombés ou cernes",
            "fever": "Fièvre",
            "fatigue": "Fatigue",
            "cough": "Toux",
            "runny_nose": "Ecoulement nasal",
            "sore_throat": "Mal de gorge",
            "aches": "Courbatures",
            "unexplained_pain": "Douleur inexpliquée",
            "high_bp": "Hypertension",
            "vomiting": "Vomissements",
            "mycn": "Amplification MYCN",
            "alk": "Mutation ALK",
            "deletion_11q": "Déletion 11q",
            "gain_17q": "Gain 17q"
        }
    }
}

# ---------------- Load model & scaler ----------------
@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        return None, None, f"Model file not found: {MODEL_PATH}"
    if not os.path.exists(SCALER_PATH):
        return None, None, f"Scaler file not found: {SCALER_PATH}"
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler, None

# Load imaging model with TensorFlow check
@st.cache_resource
def load_scan_model():
    if not TENSORFLOW_AVAILABLE:
        return None, "TensorFlow not installed, imaging model not available."
    if not os.path.exists(SCAN_MODEL_PATH):
        return None, f"Scan model file not found: {SCAN_MODEL_PATH}"
    scan_model = load_model(SCAN_MODEL_PATH)
    return scan_model, None

model, scaler, load_error = load_model_and_scaler()
scan_model, scan_load_error = load_scan_model()

if load_error:
    st.error(load_error)
    st.stop()

# ---------------- Helpers ----------------
def gender_to_numeric(g):
    return 1 if isinstance(g, str) and g.lower().startswith("m") else 0

def save_patient_row(row: dict):
    df_row = pd.DataFrame([row])
    header = not os.path.exists(PATIENTS_CSV)
    df_row.to_csv(PATIENTS_CSV, mode="a", header=header, index=False)

def load_patients():
    if os.path.exists(PATIENTS_CSV):
        try:
            return pd.read_csv(PATIENTS_CSV)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

if "patients_df" not in st.session_state:
    st.session_state["patients_df"] = load_patients()
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_scan_result" not in st.session_state:
    st.session_state["last_scan_result"] = None
    
if "run_scan" not in st.session_state:
    st.session_state["run_scan"] = False

# ---------------- Sidebar (MOVED UP - DEFINES LANG EARLY) ----------------
with st.sidebar:
    lang_display = st.selectbox("🌐 Website Language", options=["English", "Spanish", "French"], index=0)
    lang = lang_display
    t = translations[lang]

    st.markdown("---")
    st.info(t["fill_patient_note"])

    assessment_date = st.date_input(t["assessment_date"], value=datetime.now().date(), key="assessment_date")
    age = st.number_input(t["age"], min_value=0, max_value=120, value=5, step=1, key="age")
    gender = st.selectbox(t["gender"], options=[t["male"], t["female"], t["other"]], key="gender")

    st.markdown("---")
    st.markdown(f"### {t['past_patient_data']}")
    if not st.session_state["patients_df"].empty:
        st.dataframe(st.session_state["patients_df"], use_container_width=True)
        st.download_button(
            t["download_all_csv"],
            data=st.session_state["patients_df"].to_csv(index=False).encode(),
            file_name="patients.csv",
            mime="text/csv"
        )
    else:
        st.info("No past patient data yet.")

# ---------------- Main ----------------
st.title(t["title"])
st.markdown(t["disclaimer"])
st.subheader(t["nutshell_title"])
st.write(t["nutshell_text"])
st.markdown("---")

# ------ Symptoms ------
st.subheader(t["major_symptoms"])
maj_col1, maj_col2 = st.columns(2)
with maj_col1:
    s_lump = st.checkbox(t["symptom_list"]["lump"])
    s_abdominal_pain = st.checkbox(t["symptom_list"]["abdominal_pain"])
with maj_col2:
    s_weight_loss = st.checkbox(t["symptom_list"]["weight_loss"])
    s_constipation = st.checkbox(t["symptom_list"]["constipation"])

st.markdown("---")
st.subheader(t["rarer_symptoms"])
maj_col1, maj_col2 = st.columns(2)
with maj_col1:
    s_bone_pain = st.checkbox(t["symptom_list"]["bone_pain"])
with maj_col2:
    s_bulging_eyes = st.checkbox(t["symptom_list"]["bulging_eyes"])

st.markdown("---")
st.subheader(t["additional_symptoms"])
add_col1, add_col2, add_col3 = st.columns(3)
with add_col1:
    s_fever = st.checkbox(t["symptom_list"]["fever"])
    s_fatigue = st.checkbox(t["symptom_list"]["fatigue"])
    s_cough = st.checkbox(t["symptom_list"]["cough"])
with add_col2:
    s_runny = st.checkbox(t["symptom_list"]["runny_nose"])
    s_sore = st.checkbox(t["symptom_list"]["sore_throat"])
    s_aches = st.checkbox(t["symptom_list"]["aches"])
with add_col3:
    s_unexplained_pain = st.checkbox(t["symptom_list"]["unexplained_pain"])
    s_high_bp = st.checkbox(t["symptom_list"]["high_bp"])
    s_vomiting = st.checkbox(t["symptom_list"]["vomiting"])

# ------ Lab Results ------
st.markdown("---")
st.subheader(t["lab_results_title"])

genetics_not_checked = st.checkbox(
    "Not checked for genetic changes yet",
    value=False,
    help="If checked, the model will ignore MYCN, ALK, 11q, and 17q and only use symptoms."
)

lab_col1, lab_col2 = st.columns(2)
with lab_col1:
    s_mycn = st.checkbox(t["symptom_list"]["mycn"], disabled=genetics_not_checked)
    s_alk = st.checkbox(t["symptom_list"]["alk"], disabled=genetics_not_checked)
with lab_col2:
    s_11q = st.checkbox(t["symptom_list"]["deletion_11q"], disabled=genetics_not_checked)
    s_17q = st.checkbox(t["symptom_list"]["gain_17q"], disabled=genetics_not_checked)

# ---------------- Prediction button and results ----------------
st.markdown("---")
predict_clicked = st.button(t["predict_button"])
results_placeholder = st.empty()

def compute_and_store_result():

    if genetics_not_checked:
        mycn_val = 0
        alk_val = 0
        q11_val = 0
        q17_val = 0
    else:
        mycn_val = int(s_mycn)
        alk_val = int(s_alk)
        q11_val = int(s_11q)
        q17_val = int(s_17q)

    features = [
        age,
        gender_to_numeric(gender),
        int(s_lump),
        int(s_abdominal_pain),
        int(s_weight_loss),
        int(s_constipation),
        int(s_bone_pain),
        int(s_bulging_eyes),
        int(s_fever),
        int(s_fatigue),
        int(s_cough),
        int(s_runny),
        int(s_sore),
        int(s_aches),
        int(s_unexplained_pain),
        int(s_high_bp),
        int(s_vomiting),
        mycn_val,
        alk_val,
        q11_val,
        q17_val
    ]

    X = np.array([features], dtype=float)
    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)[0]
    pred = model.predict(Xs)[0]

    neuro_prob = float(probs[1])
    non_neuro_prob = float(probs[0])

    # Confidence based on predicted class
    if pred == 1:  # Neuroblastoma
        confidence = neuro_prob * 100
        prediction_text = "Neuroblastoma"
    else:
        confidence = non_neuro_prob * 100
        prediction_text = "No Neuroblastoma"

    # Risk levels
    if neuro_prob <= 0.34:
        risk_level = t["risk_low"]
        dot_color = "#2ca02c"
        suggestion = t["suggestions_low"]
    elif neuro_prob <= 0.50:
        risk_level = t["risk_mild"]
        dot_color = "#ffc107"  # yellow
        suggestion = t["suggestions_mild"]
    elif neuro_prob <= 0.74:
        risk_level = t["risk_moderate"]
        dot_color = "#f0ad4e"
        suggestion = t["suggestions_moderate"]
    else:
        risk_level = t["risk_high"]
        dot_color = "#d62728"
        suggestion = t["suggestions_high"]

    result = {
        "Date": str(assessment_date),
        "Age": age,
        "Gender": gender,
        "Prediction": pred,
        "Prediction_Text": prediction_text,
        "Probability_%": round(neuro_prob * 100, 2),
        "Confidence_%": round(confidence, 2),
        "Risk": risk_level
    }

    st.session_state["last_result"] = {
        "result": result,
        "dot_color": dot_color,
        "suggestion": suggestion,
        "confidence": confidence,
        "neuro_prob": neuro_prob
    }

if predict_clicked:
    try:
        compute_and_store_result()
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

if st.session_state.get("last_result"):
    r = st.session_state["last_result"]
    res = r["result"]
    dot_color = r["dot_color"]
    suggestion = r["suggestion"]
    confidence = r["confidence"]
    neuro_prob = r["neuro_prob"]

    with results_placeholder.container():
        st.markdown(t["risk_ref_title"])
        st.markdown(t["risk_ref_text"])
        st.markdown("---")

        st.markdown("### 🔬 Prediction Results")
        st.markdown(
            f"<span class='risk-dot' style='background:{dot_color}'></span> **{res['Risk']}**",
            unsafe_allow_html=True
        )
        st.write(f"**Prediction:** {res['Prediction_Text']}")
        st.write(f"**Probability:** {res['Probability_%']:.1f}%")

        st.markdown("**Suggestions:**")
        st.write(suggestion)

        st.markdown("**Model confidence:**")
        st.progress(int(confidence))
        st.write(f"{confidence:.2f}% confident this patient has {res['Prediction_Text'].lower()}.")


        # Download CSV
        single_df = pd.DataFrame([res])
        st.download_button(
            t["download_csv"],
            data=single_df.to_csv(index=False).encode(),
            file_name=f"assessment_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

        st.markdown("---")
        store = st.checkbox(t["store_data"], value=False)
        if store:
            save_patient_row(res)
            st.success("✅ Data stored.")
            st.session_state["patients_df"] = load_patients()

# ------ Scan Upload Section ------
st.markdown("---")
st.subheader(t["scan_section_title"])

uploaded_scan = st.file_uploader(
    t["scan_uploader_label"],
    type=["jpg", "jpeg"],
    key="scan_uploader"
)

if uploaded_scan is not None:
    st.image(uploaded_scan, caption="Uploaded scan", use_container_width=True)

    if not TENSORFLOW_AVAILABLE or scan_model is None:
        st.warning(t["scan_model_not_available"])
    else:
        if st.button(t["scan_analyze_button"], key="scan_analyze"):
            st.session_state["run_scan"] = True
        
from tensorflow.keras.applications.efficientnet import preprocess_input

if st.session_state.get("run_scan", False) and uploaded_scan is not None:
    try:
        # Load and resize image
        img = Image.open(uploaded_scan).convert("RGB")  # ensures 3 channels
        img = img.resize((224, 224))
        img_arr = np.array(img, dtype=np.float32)      # shape: (224, 224, 3)
        img_arr = np.expand_dims(img_arr, axis=0)      # shape: (1, 224, 224, 3)
        img_arr = preprocess_input(img_arr)

        # Predict
        scan_probs = scan_model.predict(img_arr)[0]
        pred_idx = int(np.argmax(scan_probs))

        # Correct class mapping
        CLASS_MAP = {0: "neuroblastoma", 1: "non_neuroblastoma"}
        prediction_text = CLASS_MAP[pred_idx]
        scan_prob_neuro = float(scan_probs[0])  # neuroblastoma probability is index 0

        # Risk logic based on neuroblastoma probability
        if scan_prob_neuro <= 0.34:
            scan_risk = t["risk_low"]
            scan_color = "#2ca02c"
            scan_suggestion = "Low imaging risk. Continue monitoring."
        elif scan_prob_neuro <= 0.50:
            scan_risk = t["risk_mild"]
            scan_color = "#ffc107"
            scan_suggestion = "Mild imaging risk. Consider follow-up."
        elif scan_prob_neuro <= 0.74:
            scan_risk = t["risk_moderate"]
            scan_color = "#f0ad4e"
            scan_suggestion = "Moderate imaging risk. Specialist review advised."
        else:
            scan_risk = t["risk_high"]
            scan_color = "#d62728"
            scan_suggestion = "High imaging risk. Urgent evaluation recommended."

        # Store results
        st.session_state["last_scan_result"] = {
            "prob_neuro": scan_prob_neuro,
            "risk": scan_risk,
            "color": scan_color,
            "suggestion": scan_suggestion,
            "prediction_text": prediction_text
        }

        # Display results
        st.markdown("### 🧠 Scan-Based Risk Assessment")
        st.markdown(
            f"<span class='risk-dot' style='background:{scan_color}'></span> **{scan_risk}**",
            unsafe_allow_html=True
        )
        st.write(f"**Prediction:** {prediction_text}")
        st.write(f"**Neuroblastoma probability:** {scan_prob_neuro * 100:.1f}%")
        st.write(scan_suggestion)

        # Reset trigger
        st.session_state["run_scan"] = False

    except Exception as e:
        st.error(f"Scan analysis error: {e}")
        st.session_state["run_scan"] = False

#----------------- Combined Result -----------------------
st.markdown("---")
final_combined = st.button("🧬 Final Combined Risk")

def compute_final_combined_result():
    if st.session_state.get("last_result") is None:
        st.warning("Please run symptom-based prediction first.")
        return

    if st.session_state.get("last_scan_result") is None:
        st.warning("Please analyze a scan before combining results.")
        return

    neuro_prob = st.session_state["last_result"]["neuro_prob"]
    scan_prob = st.session_state["last_scan_result"]["prob_neuro"]

    combined_prob = (neuro_prob + scan_prob) / 2.0

    if combined_prob <= 0.34:
        risk = t["risk_low"]
        color = "#2ca02c"
    elif combined_prob <= 0.50:
        risk = t["risk_mild"]
        color = "#ffc107"
    elif combined_prob <= 0.74:
        risk = t["risk_moderate"]
        color = "#f0ad4e"
    else:
        risk = t["risk_high"]
        color = "#d62728"

    st.markdown("### 🧬 Final Combined Risk Assessment")
    st.markdown(
        f"<span class='risk-dot' style='background:{color}'></span> **{risk}**",
        unsafe_allow_html=True
    )
    st.write(f"**Combined probability:** {combined_prob * 100:.1f}%")
    st.write(
        "*This combined result integrates symptom-based AI prediction with scan-based analysis and is experimental.*"
    )

if final_combined:
    compute_final_combined_result()

# ---------------- Feedback (above footer) ----------------
st.markdown("---")
st.subheader(t["feedback"])
feedback_text = st.text_area(
    "🗒️ Feedback (optional) — share your thoughts or report issues",
    height=140,
    placeholder="Type your feedback here..."
)
if st.button(t["submit_feedback"]):
    if feedback_text.strip():
        st.success("Thanks for your feedback!")
    else:
        st.warning("Please enter feedback before submitting.")

st.markdown("---")
st.markdown(
    "<div class='footer'>© 2025 Neuroblastoma Risk Predictor | Contact: "
    "<a href='mailto:leonj062712@gmail.com'>leonj062712@gmail.com</a></div>",
    unsafe_allow_html=True
)
