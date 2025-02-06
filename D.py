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
from scipy.stats import entropy

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

# Function to compute entropy
def calculate_entropy(y):
    _, counts = np.unique(y, return_counts=True)
    return entropy(counts, base=2)

# Function to compute Information Gain for each feature
def compute_information_gain(X, y):
    parent_entropy = calculate_entropy(y)
    info_gain = {}

    for feature in X.columns:
        threshold = X[feature].median()  # Using median as a splitting criterion
        left_split = y[X[feature] <= threshold]
        right_split = y[X[feature] > threshold]

        left_entropy = calculate_entropy(left_split)
        right_entropy = calculate_entropy(right_split)

        left_weight = len(left_split) / len(y)
        right_weight = len(right_split) / len(y)

        # Compute Information Gain
        ig = parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)
        info_gain[feature] = ig

    return info_gain

# Compute Information Gain and display it in Streamlit
st.subheader("🔍 Information Gain for Each Feature")
info_gain_values = compute_information_gain(X, y)
info_gain_df = pd.DataFrame(list(info_gain_values.items()), columns=["Feature", "Information Gain"])
info_gain_df = info_gain_df.sort_values(by="Information Gain", ascending=False)
st.write(info_gain_df)  # Display as table

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

    # Existing inputs (Removed new features)
    for col in X.columns:
        if col in label_encoders and col not in ["Processed_Food_Fast_Food", "Fruit_Veg_Intake", "Genetic_Risk_Score", "Sugar_Consumption", "Gestational_Diabetes"]:
            options = list(label_encoders[col].classes_)
            user_input[col] = st.selectbox(f"{col}", options)
            user_input[col] = label_encoders[col].transform([user_input[col]])[0]
        else:
            user_input[col] = st.number_input(f"{col}", min_value=0.0, step=0.1)

    # New Features (Moved from previous section)
    st.subheader("New Features")
    
    # 1. Processed Food, Fast Food Consumption
    processed_food = st.selectbox("Processed Food, Fast Food Consumption (More than 10 days = Yes, Less than 10 days = No)", ["Yes", "No"])
    user_input["Processed_Food_Fast_Food"] = 1 if processed_food == "Yes" else 0

    # 2. Fruit & Veg Intake (Scale input)
    fruit_veg_intake = st.slider("Fruit & Veg Intake (Scale from 1 to 10, where 1 is Low and 10 is High)", 1, 10)
    user_input["Fruit_Veg_Intake"] = fruit_veg_intake

    # 3. Genetic Risk Score
    genetic_risk = st.selectbox("Genetic Risk Score (1 to 3 = Uncles/Aunts, 4 to 6 = Parents/Grandparents)", [1, 2, 3, 4, 5, 6])
    user_input["Genetic_Risk_Score"] = genetic_risk

    # 4. Sugar Consumption (in grams)
    sugar_consumption = st.number_input("Sugar Consumption (in grams)", min_value=0.0, step=1.0)
    user_input["Sugar_Consumption"] = sugar_consumption

    # 5. Gestational Diabetes (Yes/No)
    gestational_diabetes = st.selectbox("Gestational Diabetes (Yes or No)", ["Yes", "No"])
    user_input["Gestational_Diabetes"] = 1 if gestational_diabetes == "Yes" else 0
    
    # When the user presses the button to predict
    if st.button("Predict"):
        user_df = pd.DataFrame([user_input])
        user_df_scaled = scaler.transform(user_df)
        prediction = rf_model.predict(user_df_scaled)[0]
        st.success(f"🩺 **Predicted Diabetes Type:** {prediction}")

elif menu == "Diet Recommendations":
    st.header("🍏Diet Recommendations")
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
