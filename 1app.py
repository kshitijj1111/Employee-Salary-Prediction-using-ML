import streamlit as st
from openai import OpenAI
import pandas as pd
import joblib
import os

# # --- Google Gemini API Configuration ---
# # This code securely loads the key from your secrets.toml file
# try:
#     client = OpenAI(
#         api_key=st.secrets["GOOGLE_API_KEY"],
#         base_url="https://generativelanguage.googleapis.com/v1",
#     )
#     GEMINI_API_AVAILABLE = True
# except Exception as e:
#     GEMINI_API_AVAILABLE = False
#     st.error("Google API key not found. Make sure you have created a .streamlit/secrets.toml file and added your GOOGLE_API_KEY.")


# --- Grok API Configuration ---
# Streamlit will automatically load the key from your secrets.toml file
try:
    client = OpenAI(
        api_key=st.secrets["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
    )
    GROK_API_AVAILABLE = True
except Exception as e:
    GROK_API_AVAILABLE = False
    # Display a friendly message in the app if the key is missing
    st.error("Grok API key not found. Please add it to your .streamlit/secrets.toml file.")

# ... (the rest of your app.py code remains the same) ...


# --- Load Model and Preprocessor ---
model = joblib.load('gradient_boosting_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')


# --- Gemini Interpretation Function ---
def get_gemini_interpretation(input_data, prediction):
    """
    Gets a human-readable interpretation of the prediction from Gemini.
    """
    if not GEMINI_API_AVAILABLE:
        return "Gemini API key not configured. Please add your key to the secrets file."

    # Convert prediction to a human-readable string
    prediction_text = "> $50K/year" if prediction[0] == 1 else "<= $50K/year"

    # Create a detailed prompt for Gemini
    prompt = f"""
    A machine learning model predicted an employee's salary to be {prediction_text}.
    Please provide a brief, easy-to-understand interpretation of why the model might have made this prediction.
    Focus on the most likely contributing factors from the employee's data.

    **Employee's Data:**
    - Age: {input_data['age'].iloc[0]}
    - Workclass: {input_data['workclass'].iloc[0]}
    - Education: {input_data['education'].iloc[0]}
    - Marital Status: {input_data['marital-status'].iloc[0]}
    - Occupation: {input_data['occupation'].iloc[0]}
    - Relationship: {input_data['relationship'].iloc[0]}
    - Race: {input_data['race'].iloc[0]}
    - Gender: {input_data['gender'].iloc[0]}
    - Hours per Week: {input_data['hours-per-week'].iloc[0]}
    - Native Country: {input_data['native-country'].iloc[0]}

    **Instructions for your response:**
    - Speak directly to the user.
    - Be concise (2-3 sentences).
    - Highlight 2-3 key factors that likely influenced the prediction (e.g., "Higher education levels like 'Masters' and a high number of hours worked are strong indicators of a higher salary.").
    - Do not simply repeat the input data. Provide insight.
    """

    try:
        response = client.chat.completions.create(
            model="gemini-1.5-flash", # Using a fast and capable Gemini model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred while contacting Gemini: {e}"


# --- Streamlit Page Setup ---
st.set_page_config(page_title="Employee Salary Predictor", layout="wide")
st.title("Employee Salary Predictor 💵")

st.write("""
This app predicts whether an employee's income is greater than $50K a year.
After predicting, Google's Gemini AI will provide an interpretation of the result.
""")

# Create columns for user input
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 17, 90, 35)
    workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
    education = st.selectbox("Education", ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])

with col2:
    marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
    occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
    relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])

with col3:
    race = st.selectbox("Race", ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
    gender = st.selectbox("Gender", ["Male", "Female"])
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    # Using a selectbox for common countries plus a text input for others
    native_country_options = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'England', 'Other']
    selected_country = st.selectbox("Native Country", native_country_options)
    if selected_country == 'Other':
        native_country = st.text_input("Enter Native Country")
    else:
        native_country = selected_country

# Create a button to make a prediction
if st.button("Predict Salary", type="primary"):
    # Create a DataFrame from the user input
    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'education': [education],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'gender': [gender],
        'capital-gain': [0], # Not included in UI for simplicity
        'capital-loss': [0], # Not included in UI for simplicity
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    })

    # Preprocess the input data
    input_processed = preprocessor.transform(input_data)

    # Make the prediction
    prediction = model.predict(input_processed)
    prediction_proba = model.predict_proba(input_processed)

    # --- Display the Prediction Result ---
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success(f"The predicted income is **> $50K/year** with a probability of {prediction_proba[0][1]:.2f}.")
    else:
        st.info(f"The predicted income is **<= $50K/year** with a probability of {prediction_proba[0][0]:.2f}.")

    # --- Display the Gemini Interpretation ---
    st.subheader("💡 Gemini's Interpretation")
    with st.spinner("Gemini is thinking..."):
        interpretation = get_gemini_interpretation(input_data, prediction)
        st.markdown(interpretation)
