from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tourism Package Prediction", page_icon="✈️", layout="centered")
model_path = Path(__file__).with_name("best_tourism_package_model_v1.joblib")
st.title("Tourism Package Prediction")
st.write("Enter customer details to estimate the likelihood of purchasing a travel package.")
if not model_path.exists():
    st.error("The trained model file is not in this repository yet. Run the GitHub Actions training pipeline first.")
    st.stop()
model = joblib.load(model_path)
with st.form("customer_form"):
    age = st.slider("Age", 18, 70, 30)
    contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city = st.selectbox("City Tier", [1, 2, 3])
    pitch = st.slider("Duration of Pitch (minutes)", 0, 100, 15)
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female", "Others"])
    visitors = st.slider("Number of Persons Visiting", 1, 5, 2)
    followups = st.slider("Number of Follow-ups", 1, 10, 3)
    product = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    stars = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
    marital = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
    trips = st.slider("Number of Trips", 1, 20, 3)
    passport = st.selectbox("Has Passport?", ["Yes", "No"])
    satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    car = st.selectbox("Owns a Car?", ["Yes", "No"])
    children = st.slider("Number of Children Visiting", 0, 5, 1)
    designation = st.selectbox("Designation", ["Executive", "Manager", "AVP", "VP", "Sr. Manager"])
    income = st.number_input("Monthly Income", min_value=1000.0, value=30000.0)
    submitted = st.form_submit_button("Predict")
if submitted:
    row = pd.DataFrame([{"Age":age,"TypeofContact":contact,"CityTier":city,"DurationOfPitch":pitch,"Occupation":occupation,"Gender":gender,"NumberOfPersonVisiting":visitors,"NumberOfFollowups":followups,"ProductPitched":product,"PreferredPropertyStar":stars,"MaritalStatus":marital,"NumberOfTrips":trips,"Passport":1 if passport=="Yes" else 0,"PitchSatisfactionScore":satisfaction,"OwnCar":1 if car=="Yes" else 0,"NumberOfChildrenVisiting":children,"Designation":designation,"MonthlyIncome":income}])
    probability = float(model.predict_proba(row)[0,1])
    st.metric("Purchase probability", f"{probability:.1%}")
    st.success("Likely to purchase the package." if probability >= 0.45 else "Unlikely to purchase the package.")
