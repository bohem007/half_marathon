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

# 2. UI Layout
st.set_page_config(page_title="Marathon Predictor", layout="centered")
st.title("🏃 Half-Marathon Time Predictor")
st.markdown("Introduce yourself and tell us your gender, age, and a recent 5km time.")

user_input = st.text_area("Your Introduction", 
                          placeholder="e.g., I'm a 34 year old man and my best 5k is 22 minutes.")

if st.button("Analyze & Predict"):
    if user_input:
        with st.spinner("Processing your profile..."):
            # 3. LLM Extraction
            extracted_data = extract_info_with_llm(user_input)
            
            if extracted_data:
                # 4. Validation of Missing Data
                required_fields = ['gender', 'age_category', '5km_time']
                missing = [f for f in required_fields if extracted_data.get(f) is None]
                
                if missing:
                    st.warning(f"I couldn't find your: {', '.join(missing)}. Please be more specific!")
                    st.json(extracted_data) # Show what was found
                else:
                    st.success("Data successfully extracted!")
                    st.write(f"**Extracted Profile:** Gender: {'Female' if extracted_data['gender']==1 else 'Male'}, "
                             f"Age Category: {extracted_data['age_category']}")
                    
                    age_cat = extracted_data['age_category']
                    if age_cat < 20:
                        st.info("You're quite young! Keep up the good work and focus on building a strong aerobic base.")   
                    elif age_cat >= 90:
                        st.warning("You're in an advanced age category. Consider consulting with a healthcare professional before starting a training program.")    
                    else:
                        
                        # 5. Model Inference
                        model = get_pipeline()
                        
                        # Prepare input for PyCaret (must match training feature names)
                        input_df = pd.DataFrame([{
                            'gender': extracted_data['gender'],
                            'age_category': extracted_data['age_category'],
                            '5km_time': extracted_data['5km_time']
                        }])
                        
                        predictions = predict_model(model, data=input_df)
                        pred_seconds = predictions['prediction_label'][0]
                        
                        # Format seconds to HH:MM:SS
                        formatted_time = str(pd.to_timedelta(pred_seconds, unit='s')).split('.')[0]
                        formatted_time = formatted_time[7:]  # Remove the "0 days " prefix
                        
                        st.metric("Predicted Half-Marathon Time", formatted_time)
                        
                        if extracted_data.get('5km_time'):
                            st.info(f"Your 5km time ({extracted_data['5km_time']}s) was noted for future model improvements.")
    else:
        st.error("Please enter some text first.")