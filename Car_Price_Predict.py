#Application to predict Price for used cars modeled on OLX data.

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Define the preprocess_data function
def preprocess_data(cardf):
    numcols = cardf.select_dtypes(include=np.number)
    objcols = cardf.select_dtypes(include=['object'])

    # Fill missing numerical data with median
    for col in numcols.columns:
        numcols[col] = numcols[col].fillna(numcols[col].median())

    if 'Year' in numcols.columns:
        numcols['age'] = 2024 - numcols['Year']
        numcols = numcols.drop(['Year'], axis=1)

    # One-Hot Encode categorical variables
    objcols_dummy = pd.get_dummies(objcols)

    # Combine numerical and categorical features
    cardf_final = pd.concat([numcols, objcols_dummy], axis=1)

    return cardf_final, objcols_dummy.columns

# Sidebar for File Upload
st.sidebar.title("Used Car Price Prediction")
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])  # File upload moved above

# Sidebar Navigation
app_mode = st.sidebar.selectbox("Choose an Option", ["EDA", "Modeling", "Prediction"])

# Load Dataset from Uploaded File
@st.cache
def load_data(file):
    return pd.read_excel(file)

if uploaded_file:
    cardf = load_data(uploaded_file)

    if app_mode == "EDA":
        st.title("Exploratory Data Analysis (EDA)")

        # Display the dataset
        st.write("### Uploaded Dataset")
        st.write(cardf)

        # Descriptive Statistics
        st.header("Descriptive Statistics")
        st.write("### Numerical Data")
        st.write(cardf.describe())

        st.write("### Categorical Data")
        categorical_cols = cardf.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            st.write(f"**{col}** Value Counts:")
            st.write(cardf[col].value_counts())

        # Data Visualization
        st.header("Data Visualization")
        visualization_type = st.selectbox("Select Visualization Type", ["Histogram", "Scatter Plot", "Bar Chart", "Correlation Heatmap"])

        if visualization_type == "Histogram":
            selected_column = st.selectbox("Select Column for Histogram", cardf.columns)
            plt.figure(figsize=(10, 6))
            sns.histplot(data=cardf, x=selected_column, kde=True, bins=30, color="blue")
            st.pyplot(plt)

        elif visualization_type == "Scatter Plot":
            x_column = st.selectbox("Select X-axis", cardf.columns)
            y_column = st.selectbox("Select Y-axis", cardf.columns)
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=cardf, x=x_column, y=y_column)
            st.pyplot(plt)

        elif visualization_type == "Bar Chart":
            selected_column = st.selectbox("Select Column for Bar Chart", categorical_cols)
            plt.figure(figsize=(10, 6))
            cardf[selected_column].value_counts().plot(kind='bar', color='orange')
            plt.title(f"Bar Chart of {selected_column}")
            st.pyplot(plt)

        elif visualization_type == "Correlation Heatmap":
            st.write("### Correlation Heatmap for Numerical Features")
            plt.figure(figsize=(12, 8))
            correlation = cardf.select_dtypes(include=np.number).corr()
            sns.heatmap(correlation, annot=True, cmap="plasma", fmt=".2f", linewidths=0.5)
            st.pyplot(plt)

    elif app_mode == "Modeling":
        st.title("Model Training and Evaluation")

        # Preprocess the dataset for modeling
        cardf_final, train_columns = preprocess_data(cardf)

        # Split the data
        if 'Price' not in cardf_final.columns:
            st.error("The dataset must contain a 'Price' column for modeling.")
        else:
            X = cardf_final.drop(['Price'], axis=1)
            y = cardf_final['Price']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Sidebar for Model Selection
            model_name = st.sidebar.selectbox("Select Model", ["Linear Regression", "Decision Tree", "Random Forest"])

            if model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "Decision Tree":
                model = DecisionTreeRegressor(random_state=42)
            elif model_name == "Random Forest":
                model = RandomForestRegressor(random_state=42)

            # Train and Evaluate the Model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            st.write(f"### Model: {model_name}")
            st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
            st.write(f"**R² Score:** {r2:.2f}")

            # Feature Importance (for Tree-based models)
            if model_name in ["Decision Tree", "Random Forest"]:
                feature_importances = pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": model.feature_importances_
                }).sort_values(by="Importance", ascending=False)

                st.write("### Feature Importances")
                st.write(feature_importances)

    elif app_mode == "Prediction":
        st.title("Car Price Prediction")

        # Preprocess the dataset for prediction
        cardf_final, train_columns = preprocess_data(cardf)

        # Train the model (using Linear Regression for simplicity)
        if 'Price' not in cardf_final.columns:
            st.error("The dataset must contain a 'Price' column for prediction.")
        else:
            x_train = cardf_final.drop(['Price'], axis=1)
            y_train = cardf_final['Price']
            model = LinearRegression().fit(x_train, y_train)

            # Collect User Inputs
            st.subheader("Enter Car Details for Prediction")

            location = st.selectbox("Select Location", cardf['Location'].unique())
            seats = st.selectbox("Select Number of Seats", cardf['Seats'].dropna().unique())
            fuel_type = st.selectbox("Select Fuel Type", cardf['Fuel_Type'].unique())
            transmission = st.selectbox("Select Transmission Type", cardf['Transmission'].unique())
            owner_type = st.selectbox("Select Owner Type", cardf['Owner_Type'].unique())
            age = st.number_input("Enter Age of Car", min_value=0, max_value=50, step=1)
            mileage = st.number_input("Enter Mileage (in kmpl)", min_value=0.0, step=0.1)
            engine = st.number_input("Enter Engine Size (in CC)", min_value=0, max_value=5000, step=10)
            power = st.number_input("Enter Power (in bhp)", min_value=0, max_value=1000, step=5)
            mileage_kms = st.number_input("Enter Kilometers Driven (in km)", min_value=0, step=1000)

            # Process user input
            user_input = pd.DataFrame({
                "age": [age],
                "Mileage": [mileage],
                "Engine": [engine],
                "Power": [power],
                "Kilometers_Driven": [mileage_kms],
                "Location": [location],
                "Seats": [seats],
                "Fuel_Type": [fuel_type],
                "Transmission": [transmission],
                "Owner_Type": [owner_type]
            })

            # One-hot encode user input
            user_input = pd.get_dummies(user_input)
            for col in x_train.columns:
                if col not in user_input:
                    user_input[col] = 0

            user_input = user_input[x_train.columns]  # Align column order with training data

            scaler = StandardScaler().fit(x_train)
            user_input_scaled = scaler.transform(user_input)

            # Predict the price
            if st.button("Predict Price"):
                try:
                    predicted_price = model.predict(user_input_scaled)
                    st.write(f"Predicted Car Price: ₹{predicted_price[0]:,.2f}")
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
else:
    st.warning("Please upload an Excel file to proceed.")
