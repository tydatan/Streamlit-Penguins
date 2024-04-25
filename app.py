import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('bmi.csv')
    return df

# Main function
def main():
    st.set_page_config(page_title='BMI Classification', page_icon=":hospital:", layout='centered')

    # Custom light color palette with shades of purples and blues
    custom_palette = ['#A05195', '#D45087', '#F95D6A', '#FF7C43', '#a17bbd']

    # Custom CSS styles
    st.markdown(
        f"""
        <style>
            .reportview-container .main .block-container{{
                max-width: 1000px;
                padding-top: 1rem;
                padding-right: 1rem;
                padding-left: 1rem;
                padding-bottom: 1rem;
                background-color: #11151A; /* Change background color */
            }}
            .sidebar .sidebar-content {{
                background-color: #F7CAC9;
            }}
            .sidebar .sidebar-content .sidebar-collapse-control {{
                background-color: #F7CAC9;
            }}
            .sidebar .sidebar-content .markdown-text-container {{
                color: #6B5B95;
            }}
            .element-container .stMultiSelect, .element-container .stNumberInput, .element-container .stSelectbox, .element-container .stTextInput {{
                color: #6B5B95;
                background-color: #F7CAC9;
            }}
            .css-17eq0hr {{
                color: #6B5B95;
                background-color: #A05195;
            }}
            .stButton>button {{
                color: #6B5B95;
                background-color: #A05195;
            }}
            .css-1t42vg8 {{
                color: #6B5B95;
                background-color: #A05195;
            }}
            .stDataFrame>div>div>div>div>div>div>div>div {{
                color: #6B5B95;
                background-color: #F7CAC9;
            }}
            .stDataFrame>div>div>div>div>div>div>div>div>div>div {{
                color: #6B5B95;
                background-color: #F7CAC9;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('BMI Classification')

    st.write("""
    Welcome to the BMI Classification web app! This app predicts the BMI classification based on the given input features such as Age, Height, Weight, and BMI.
    """)

    st.write("""
    ### Instructions:
    1. Use the sliders on the left to adjust feature values for prediction.
    2. After adjusting the values, the predictions will be displayed below.
    3. You can also view the raw data and data visualizations using the checkboxes and subheaders respectively.
    """)

    # Load data
    df = load_data()

    # Data preprocessing
    labelencoder = LabelEncoder()
    df['BmiClass'] = labelencoder.fit_transform(df['BmiClass'])

    X = df.drop('BmiClass', axis=1)
    y = df['BmiClass']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sliders for adjusting feature values
    st.sidebar.subheader('Adjust values for prediction')
    age = st.sidebar.slider('Age', min_value=10, max_value=100, value=30)
    height = st.sidebar.slider('Height (m)', min_value=1.0, max_value=2.5, value=1.7, step=0.01)
    weight = st.sidebar.slider('Weight (kg)', min_value=30, max_value=200, value=70)
    bmi = weight / (height ** 2)

    # Predictions
    st.subheader('Predictions')

    # XGBoost model
    st.subheader('XGBoost Model')
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)
    xgb_prediction = xgb_model.predict_proba([[age, height, weight, bmi]])
    xgb_prediction_label = labelencoder.inverse_transform(np.argmax(xgb_prediction, axis=1))
    st.write("XGBoost Prediction:", xgb_prediction_label[0])

    # Pie chart for XGBoost probabilities
    st.subheader('XGBoost Probabilities Pie Chart')
    labels = labelencoder.classes_
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(xgb_prediction[0], labels=labels, autopct='%1.1f%%', startangle=140, colors=custom_palette)
    ax.axis('equal')
    ax.set_facecolor('#20202b')  # Change background color of the plot
    st.pyplot(fig)


    # Random Forest model
    st.subheader('Random Forest Model')
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_prediction = rf_model.predict_proba([[age, height, weight, bmi]])
    rf_prediction_label = labelencoder.inverse_transform(np.argmax(rf_prediction, axis=1))
    st.write("Random Forest Prediction:", rf_prediction_label[0])

    # Pie chart for Random Forest probabilities
    st.subheader('Random Forest Probabilities Pie Chart')
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(rf_prediction[0], labels=labels, autopct='%1.1f%%', startangle=140, colors=custom_palette)
    ax.axis('equal')
    ax.set_facecolor('#0e1117')  # Change background color of the plot
    st.pyplot(fig)

    # Accuracy
    st.subheader('Model Accuracy')
    xgb_accuracy = accuracy_score(y_test, xgb_model.predict(X_test))
    rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
    st.write("XGBoost Accuracy:", xgb_accuracy)
    st.write("Random Forest Accuracy:", rf_accuracy)





    # Visualizations
    st.subheader('Data Visualizations')

    st.write(""" Below is a breakdown of the dataset used to train the model. Explore the contents and visualization """)

    # Display data
    if st.checkbox('Show raw data'):
        st.subheader('Raw Data')
        st.write(df)

    # Countplot of BmiClass
    st.subheader('Countplot of BmiClass')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='BmiClass', data=df, palette=custom_palette)
    ax.set_title('Countplot of BmiClass')
    ax.set_xlabel('BmiClass')
    ax.set_ylabel('Count')
    ax.set_facecolor('#20202b')
    st.pyplot(fig)

    # Distribution of Numerical Variables
    st.subheader('Distribution of Numerical Variables')
    numerical_cols = ['Age', 'Height', 'Weight', 'Bmi']
    for col in numerical_cols:
        st.write(f'Distribution of {col}')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df[col], bins=20, kde=True, edgecolor='black', ax=ax, color=custom_palette[0])
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.set_facecolor('#20202b')
        st.pyplot(fig)

    # Correlation Matrix
    st.subheader('Correlation Matrix')
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_facecolor('#0e1117')  # Change background color of the plot
    st.pyplot(fig)

if __name__ == '__main__':
    main()
