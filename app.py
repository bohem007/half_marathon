import os
from dotenv import load_dotenv
from pathlib import Path
from langfuse import observe
from langfuse.openai import OpenAI

import streamlit as st
import pandas as pd
import json
import boto3
from botocore.exceptions import ClientError
import joblib

from pycaret.regression import predict_model


load_dotenv()

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

s3 = boto3.client(
    "s3",
)
BUCKET_NAME = 'bohemspace'
MODEL_NAME = 'best_marathon_pipeline.pkl'


@st.cache_resource
def get_pipeline():
    try:
        s3.download_file(BUCKET_NAME, MODEL_NAME, MODEL_NAME)
        model = joblib.load(MODEL_NAME)
        os.remove(MODEL_NAME)  # Clean up the local file after loading
        return model
    except ClientError as e:
        st.error(f"Error loading model from S3: {e}")
        return None
    
@observe()
def extract_info_with_llm(user_text):
    """
    Uses OpenAI to extract gender, age, and 5km time from natural language.
    Returns a dictionary.
    """
    prompt = f"""
    Extract the following information from the user's introduction in JSON format:
    - gender: (int: 0 for male/man, 1 for female/woman)
    - age: (int)
    - age_category: (int: the age floored to the nearest 10, e.g., 34 -> 30, 45 -> 40)
    - 5km_time: (int: convert their 5km time to total seconds)

    If a value is missing, set it to null.
    
    User text: "{user_text}"
    """
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o", 
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"LLM Extraction Error: {e}")
        return None


# ===============================
# 🌍 LANGUAGE CONFIGURATION
# ===============================

LANGUAGES = {
    "English": "en",
    "Polski": "pl",
    "Deutsch": "de"
}

TRANSLATIONS = {
    "en": {
        "title": "🏃 Half-Marathon Time Predictor",
        "intro": "Introduce yourself and tell us your gender, age, and a recent 5km time.",
        "input_label": "Your Introduction",
        "placeholder": "e.g., I'm a 34 year old man and my best 5k is 22 minutes.",
        "button": "Analyze & Predict",
        "processing": "Processing your profile...",
        "missing": "I couldn't find your:",
        "be_specific": "Please be more specific!",
        "extracted": "Data successfully extracted!",
        "profile": "Extracted Profile",
        "young": "You're quite young! Keep up the good work and focus on building a strong aerobic base.",
        "senior": "You're in an advanced age category. Consider consulting with a healthcare professional before starting a training program.",
        "prediction": "Predicted Half-Marathon Time",
        "empty": "Please enter some text first.",
        "gender_male": "male",
        "gender_female": "female",
        "age_category": "Age Category: ",
        "gender": "gender",
        "age": "age",
        "5km_time": "5km_time",
        "notice1": "Your 5km time ",
        "notice2": " was noted for future model improvements."
    },
    "pl": {
        "title": "🏃 Predyktor Czasu Półmaratonu",
        "intro": "Przedstaw się i podaj swoją płeć, wiek oraz ostatni czas na 5 km.",
        "input_label": "Twoje Wprowadzenie",
        "placeholder": "np. Mam 34 lata, jestem mężczyzną, a moje najlepsze 5 km to 22 minuty.",
        "button": "Analizuj i Przewiduj",
        "processing": "Analizuję Twój profil...",
        "missing": "Nie mogłem znaleźć:",
        "be_specific": "Podaj więcej szczegółów!",
        "extracted": "Dane zostały poprawnie wyodrębnione!",
        "profile": "Wyodrębniony Profil",
        "young": "Jesteś młody! Kontynuuj dobrą pracę i buduj solidną bazę tlenową.",
        "senior": "Jesteś w zaawansowanej kategorii wiekowej. Skonsultuj się z lekarzem przed rozpoczęciem programu treningowego.",
        "prediction": "Przewidywany Czas Półmaratonu",
        "empty": "Najpierw wpisz tekst.",
        "gender_male": "mężczyzna",
        "gender_female": "kobieta",
        "age_category": "Kategoria wieku: ",
        "gender": "płeć",
        "age": "wiek",
        "5km_time": "5km_czas",
        "notice1": "Twój czas na 5 km ",
        "notice2": " został zanotowany do przyszłych ulepszeń modelu."
    },
    "de": {
        "title": "🏃 Halbmarathon-Zeitvorhersage",
        "intro": "Stelle dich vor und gib dein Geschlecht, Alter und deine letzte 5-km-Zeit an.",
        "input_label": "Deine Vorstellung",
        "placeholder": "z.B. Ich bin 34 Jahre alt, männlich, und meine beste 5-km-Zeit liegt bei 22 Minuten.",
        "button": "Analysieren & Vorhersagen",
        "processing": "Profil wird verarbeitet...",
        "missing": "Folgende Angaben fehlen:",
        "be_specific": "Bitte sei genauer!",
        "extracted": "Daten erfolgreich extrahiert!",
        "profile": "Extrahiertes Profil",
        "young": "Du bist noch jung! Baue weiterhin eine starke aerobe Basis auf.",
        "senior": "Du befindest dich in einer höheren Altersklasse. Bitte konsultiere einen Arzt vor Trainingsbeginn.",
        "prediction": "Vorhergesagte Halbmarathonzeit",
        "empty": "Bitte gib zuerst einen Text ein.",
        "gender_male": "männlich",
        "gender_female": "weiblich",
        "age_category": "Alterskategorie: ",
        "gender": "Geschlecht",
        "age": "Alter",
        "5km_time": "5km_Zeit",
        "notice1": "Deine 5-km-Zeit ", 
        "notice2": " wurde für zukünftige Modellverbesserungen notiert."
    }
}



# ===============================
# 🚀 FLAG SELECTOR (PRODUCTION SAFE)
# ===============================

if "language" not in st.session_state:
    st.session_state.language = "en"
    

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("EN"):
        st.session_state.language = "en"
        
with col2:
    if st.button("🇵🇱"):
        st.session_state.language = "pl"
        
with col3:
    if st.button("🇩🇪"):
        st.session_state.language = "de"
 

lang_code = st.session_state.language

def t(key):
    return TRANSLATIONS[lang_code][key]


#'Female' if extracted_data['gender']==1 else 'Male'
def gender_str(gender_val):
    if gender_val == 1:
        return t("gender_female")
    else:
        return t("gender_male")

# ===============================
# 🎨 UI LAYOUT
# ===============================

st.set_page_config(page_title="Marathon Predictor", layout="centered")

st.title(t("title"))
st.markdown(t("intro"))

user_input = st.text_area(
    t("input_label"),
    placeholder=t("placeholder")
)

if st.button(t("button")):

    if user_input:

        with st.spinner(t("processing")):

            extracted_data = extract_info_with_llm(user_input)

            if extracted_data:

                required_fields = ['gender', 'age_category', '5km_time']
                missing = [t(f) for f in required_fields if extracted_data.get(f) is None]

                if missing:
                    st.warning(f"{t('missing')} {', '.join(missing)}. {t('be_specific')}")
                    # st.json(extracted_data)
                else:

                    st.success(t("extracted"))

                    st.write(
                        f"**{t('profile')}:** "
                        f"{gender_str(extracted_data['gender'])}, "
                        f"{t('age_category')}{extracted_data['age_category']}"
                    )

                    age_cat = extracted_data['age_category']

                    if age_cat < 20:
                        st.info(t("young"))
                    elif age_cat >= 90:
                        st.warning(t("senior"))
                    else:

                        model = get_pipeline()

                        input_df = pd.DataFrame([{
                            'gender': extracted_data['gender'],
                            'age_category': extracted_data['age_category'],
                            '5km_time': extracted_data['5km_time']
                        }])

                        predictions = predict_model(model, data=input_df)
                        pred_seconds = predictions['prediction_label'][0]

                        formatted_time = str(pd.to_timedelta(pred_seconds, unit='s')).split('.')[0]
                        formatted_time = formatted_time[7:]

                        st.metric(t("prediction"), formatted_time)
                        
                        if extracted_data.get('5km_time'):
                            st.info(f"{t('notice1')}{extracted_data['5km_time']}s{t('notice2')}")

    else:
        st.error(t("empty"))