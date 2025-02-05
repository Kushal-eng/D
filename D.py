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
columns_to_remove = ["Ethnicity", "Mental Health Condition", "Alcohol Consumption", "Alcohol Drinks/Week", "Hypertension"]
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
menu = st.sidebar.radio("Select an Option:", ["Diabetes Prediction", "Workout Progress Tracker", "AI-Based Diet Recommendations"])

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

# --- 🏋️‍♂️ Workout Progress Tracker ---
elif menu == "Workout Progress Tracker":
    st.header("🏋️‍♂️ Track Your Workout Progress")

    # User Inputs
    weekly_workout_hours = st.number_input("Enter your total workout hours this week:", min_value=0.0, step=0.5)
    sedentary_hours = st.number_input("Enter your daily sedentary hours:", min_value=0.0, step=0.5)

    # Progress Assessment
    if st.button("Track Progress"):
        if weekly_workout_hours >= 5 and sedentary_hours < 6:
            st.success("✅ Great job! Your activity level is excellent. Keep it up!")
        elif weekly_workout_hours >= 3:
            st.warning("⚠️ Good effort, but try to increase your activity.")
        else:
            st.error("🚨 You need to work out more! Reduce sedentary time for better health.")

# --- 🍎 AI-Based Diet Recommendations ---
elif menu == "AI-Based Diet Recommendations":
    st.header("🍎 Get AI-Powered Diet Recommendations")

    # User Inputs
    bmi = st.number_input("Enter your BMI:", min_value=10.0, step=0.1)
    calorie_intake = st.number_input("Enter your daily calorie intake:", min_value=500, step=100)
    diet_type = st.selectbox("Choose your diet type:", ["Balanced", "High-Protein", "Low-Carb", "Vegan"])

    # AI Diet Suggestion
    if st.button("Get Diet Plan"):
        if bmi > 25:
            st.warning("⚠️ You might need a weight-loss diet. Consider reducing calorie intake and increasing fiber & protein.")
        elif calorie_intake < 1500:
            st.warning("⚠️ Your calorie intake is low. Ensure you're getting enough nutrients.")
        else:
            st.success("✅ Your diet seems well-balanced. Keep maintaining a healthy lifestyle!")

        st.write("🔹 **Recommended Foods:**")
        if diet_type == "Balanced":
            st.write("🥗 Include lean protein, whole grains, and vegetables.")
        elif diet_type == "High-Protein":
            st.write("🍗 Add chicken, fish, eggs, and legumes.")
        elif diet_type == "Low-Carb":
            st.write("🥑 Focus on healthy fats and proteins, and limit processed carbs.")
        elif diet_type == "Vegan":
            st.write("🌱 Eat plant-based proteins, nuts, seeds, and whole grains.")

