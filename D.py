import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("d.csv")
    return df

df = load_data()

# Remove unnecessary columns
columns_to_remove = ["Ethnicity", "Mental Health Condition", "Alcohol Consumption", "Alcohol Drinks/Week", "Hypertension", "Sedentary Hours"]
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

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train Decision Tree Model
st.sidebar.header("🔍 Decision Tree Model")
selected_features = st.sidebar.multiselect("Select Features for Decision Tree", X.columns.tolist(), default=X.columns.tolist())
if selected_features:
    X_selected = df[selected_features]
    X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    dt_model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
    dt_model.fit(X_train_dt, y_train_dt)
    y_pred_dt = dt_model.predict(X_test_dt)
    
    # Performance Metrics
    accuracy = accuracy_score(y_test_dt, y_pred_dt)
    precision = precision_score(y_test_dt, y_pred_dt, average='weighted', zero_division=0)
    recall = recall_score(y_test_dt, y_pred_dt, average='weighted', zero_division=0)
    
    st.sidebar.subheader("📊 Decision Tree Performance")
    st.sidebar.write(f"*Accuracy:* {accuracy:.2f}")
    st.sidebar.write(f"*Precision:* {precision:.2f}")
    st.sidebar.write(f"*Recall:* {recall:.2f}")
    
    st.subheader("📌 Decision Tree Visualization")
    plt.figure(figsize=(12, 6))
    plot_tree(dt_model, feature_names=selected_features, class_names=["Type 1", "Type 2", "Prediabetes"], filled=True)
    st.pyplot(plt)
    
    # Feature Importance
    st.subheader("💡 Feature Importance")
    feature_importance = pd.DataFrame({"Feature": selected_features, "Importance": dt_model.feature_importances_}).sort_values(by="Importance", ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="viridis")
    plt.title("Feature Importance in Decision Tree")
    st.pyplot(plt)

# Streamlit Dashboard
st.title("🏥 Diabetes Predictor Dashboard")

# Sidebar Navigation
menu = st.sidebar.radio("Select an Option:", ["Diabetes Prediction", "Diet Recommendations"])

if menu == "Diabetes Prediction":
    st.header("🔍 Predict Your Diabetes Type")
    user_input = {}
    for col in X.columns:
        if col in label_encoders:
            options = list(label_encoders[col].classes_)
            user_input[col] = st.selectbox(f"{col}", options)
            user_input[col] = label_encoders[col].transform([user_input[col]])[0]
        else:
            user_input[col] = st.number_input(f"{col}", min_value=0.0, step=0.1)
    
    if st.button("Predict"):
        user_df = pd.DataFrame([user_input])
        user_df_scaled = scaler.transform(user_df)
        prediction = rf_model.predict(user_df_scaled)[0]
        st.success(f"🩺 **Predicted Diabetes Type:** {prediction}")

elif menu == "Diet Recommendations":
    st.header("🍏 Get AI-Powered Diet Recommendations")
    bmi = st.number_input("Enter your BMI:", min_value=10.0, step=0.1)
    calorie_intake = st.number_input("Enter your daily calorie intake:", min_value=500, step=100)
    diet_type = st.selectbox("Choose your diet type:", ["Balanced", "High-Protein", "Low-Carb", "Vegan"])
    
    diet_plan = {
        "Balanced": {"Breakfast": "Oatmeal with nuts and fruit", "Lunch": "Grilled chicken with quinoa", "Dinner": "Salmon with roasted sweet potatoes"},
        "High-Protein": {"Breakfast": "Scrambled eggs with toast", "Lunch": "Grilled steak with brown rice", "Dinner": "Lentil soup with chicken"},
        "Low-Carb": {"Breakfast": "Avocado and eggs", "Lunch": "Grilled salmon with cauliflower rice", "Dinner": "Zucchini noodles with pesto"},
        "Vegan": {"Breakfast": "Smoothie with almond milk", "Lunch": "Chickpea salad with quinoa", "Dinner": "Stir-fried tofu with vegetables"}
    }
    
    if st.button("Get Diet Plan"):
        st.success("✅ Here's your recommended diet plan!")
        st.write(f"**Breakfast:** {diet_plan[diet_type]['Breakfast']}")
        st.write(f"**Lunch:** {diet_plan[diet_type]['Lunch']}")
        st.write(f"**Dinner:** {diet_plan[diet_type]['Dinner']}")
