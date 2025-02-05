import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("d.csv") 
    return df

df = load_data()

# Remove unnecessary columns
columns_to_remove = ["Ethnicity", "Mental Health Condition", "Alcohol Consumption", "Alcohol Drinks/Week", "Hypertension","Sedentary Hours"]
df = df.drop(columns=columns_to_remove)

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=["Diabetes Type"])  # Replace with actual target column name
y = df["Diabetes Type"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit Dashboard
st.title("🏥 Diabetes Predictor Dashboard")

# Sidebar Navigation
menu = st.sidebar.radio("Select an Option:", ["Diabetes Prediction", "AI-Based Diet Recommendations"])

# --- 🩺 Diabetes Prediction ---
if menu == "Diabetes Prediction":
    st.header("🔍 Predict Your Diabetes Type")

    # User Input Form
    user_input = {}
    for col in X.columns:
        if col in label_encoders:
            options = list(label_encoders[col].classes_)
            user_input[col] = st.selectbox(f"{col}", options)
            user_input[col] = label_encoders[col].transform([user_input[col]])[0]
        else:
            user_input[col] = st.number_input(f"{col}", min_value=0.0, step=0.1)

    # Prediction Button
    if st.button("Predict"):
        user_df = pd.DataFrame([user_input])
        user_df_scaled = scaler.transform(user_df)
        prediction = model.predict(user_df_scaled)[0]
        st.success(f"🩺 **Predicted Diabetes Type:** {prediction}")

# --- 🍎 AI-Based Diet Recommendations ---
elif menu == "AI-Based Diet Recommendations":
    st.header("🍎 Get AI-Powered Diet Recommendations")

    # User Inputs
    bmi = st.number_input("Enter your BMI:", min_value=10.0, step=0.1)
    calorie_intake = st.number_input("Enter your daily calorie intake:", min_value=500, step=100)
    diet_type = st.selectbox("Choose your diet type:", ["Balanced", "High-Protein", "Low-Carb", "Vegan"])

    diet_plan = {
        "Balanced": {
            "Breakfast": "Oatmeal with nuts and fruit (350 kcal)",
            "Lunch": "Grilled chicken with quinoa and vegetables (600 kcal)",
            "Dinner": "Salmon with roasted sweet potatoes and salad (500 kcal)"
        },
        "High-Protein": {
            "Breakfast": "Scrambled eggs with whole-grain toast (400 kcal)",
            "Lunch": "Grilled steak with brown rice and steamed veggies (650 kcal)",
            "Dinner": "Lentil soup with grilled chicken (550 kcal)"
        },
        "Low-Carb": {
            "Breakfast": "Avocado and eggs on spinach (300 kcal)",
            "Lunch": "Grilled salmon with cauliflower rice (550 kcal)",
            "Dinner": "Zucchini noodles with pesto and grilled chicken (500 kcal)"
        },
        "Vegan": {
            "Breakfast": "Smoothie with almond milk, banana, and peanut butter (350 kcal)",
            "Lunch": "Chickpea salad with quinoa and avocado (600 kcal)",
            "Dinner": "Stir-fried tofu with vegetables and brown rice (500 kcal)"
        }
    }

    if st.button("Get Diet Plan"):
        st.success("✅ Here's your recommended diet plan!")
        st.write(f"**Breakfast:** {diet_plan[diet_type]['Breakfast']}")
        st.write(f"**Lunch:** {diet_plan[diet_type]['Lunch']}")
        st.write(f"**Dinner:** {diet_plan[diet_type]['Dinner']}")

       
