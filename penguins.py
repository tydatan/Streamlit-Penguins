import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('penguins.csv')
    # Drop rows with missing values
    df = df.dropna()
    return df


# Main function
def main():
    st.set_page_config(page_title='Penguin Classification', page_icon="🐧", layout='centered')

    st.title('PINGU - Penguin Classification')

    st.write("""
    Welcome to PINGU, the Penguin Classification web app! This app predicts the penguin species and gender based on the given input features such as bill length, bill depth, flipper length, body mass, and other attributes.
    """)
    st.subheader('')
    st.subheader('')
    st.subheader('')

    # Load data and drop unnecessary columns
    df = load_data()
    df_clean = df.drop(['island', 'species', 'Unnamed: 0', 'year'], axis=1)

    # Data preprocessing
    labelencoder_species = LabelEncoder()
    labelencoder_sex = LabelEncoder()
    df_clean['species'] = labelencoder_species.fit_transform(df['species'])
    df_clean['sex'] = labelencoder_sex.fit_transform(df['sex'])

    X = df_clean.drop(['species', 'sex'], axis=1)
    y_species = df_clean['species']
    y_sex = df_clean['sex']

    # Train-test split
    X_train, X_test, y_train_species, y_test_species = train_test_split(X, y_species, test_size=0.2, random_state=42)
    X_train, X_test, y_train_sex, y_test_sex = train_test_split(X, y_sex, test_size=0.2, random_state=42)

    # Sliders for adjusting feature values
    st.sidebar.subheader('Adjust values for prediction')
    bill_length_mm = st.sidebar.slider('Bill Length (mm)', min_value=30.0, max_value=60.0, value=45.0, step=0.1)
    bill_depth_mm = st.sidebar.slider('Bill Depth (mm)', min_value=10.0, max_value=25.0, value=15.0, step=0.1)
    flipper_length_mm = st.sidebar.slider('Flipper Length (mm)', min_value=150.0, max_value=240.0, value=200.0,
                                          step=1.0)
    body_mass_g = st.sidebar.slider('Body Mass (g)', min_value=2000, max_value=7000, value=3500, step=50)
    st.subheader('')
    st.subheader('')

    # Predictions
    st.subheader('Species Classifier:')

    # Random Forest model for species classification
    rf_model_species = RandomForestClassifier()
    rf_model_species.fit(X_train, y_train_species)
    rf_prediction_species = rf_model_species.predict_proba(
        [[bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g]])

    # Transform predicted probabilities to labels (species)
    predicted_labels_species = labelencoder_species.inverse_transform(
        np.argsort(rf_prediction_species, axis=1)[:, ::-1][0][:3])  # Top 3 predicted species

    # Create a pie chart for species prediction
    st.subheader('')
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(rf_prediction_species[0], labels=predicted_labels_species, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

    # Display prediction
    st.write("Random Forest Prediction (Top 3 species):")
    st.write(predicted_labels_species)

    # Accuracy
    st.subheader('')
    rf_accuracy_species = accuracy_score(y_test_species, rf_model_species.predict(X_test))
    st.write("Accuracy:", rf_accuracy_species)
    st.write("")

    # Predictions for Gender Classifier
    st.subheader('Gender Classifier:')

    # Random Forest model for gender classification
    rf_model_sex = RandomForestClassifier()
    rf_model_sex.fit(X_train, y_train_sex)
    rf_prediction_sex = rf_model_sex.predict_proba([[bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g]])

    # Transform predicted probabilities to labels (male or female)
    predicted_labels_sex = labelencoder_sex.inverse_transform(
        [0, 1])  # Assuming 0 represents male and 1 represents female

    # Create a pie chart for gender prediction
    st.subheader('')
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(rf_prediction_sex[0], labels=predicted_labels_sex, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

    # Display prediction for gender
    st.write(" Gender Prediction:", predicted_labels_sex[np.argmax(rf_prediction_sex)])

    # Accuracy for gender classification
    rf_accuracy_sex = accuracy_score(y_test_sex, rf_model_sex.predict(X_test))
    st.write("Accuracy:", rf_accuracy_sex)
    st.write("")
    st.write("")
    st.write("")

    # Visualizations
    st.subheader('Data Visualizations')

    st.write(
        """ Below is a breakdown of the dataset used to train the model. Explore the contents and visualizations of our data. """)
    st.subheader('')

    # Display data
    if st.checkbox('Show raw data'):
        st.subheader('Raw Data')
        st.write(df)

    # Countplot of Species
    st.subheader('')
    st.subheader('')
    st.subheader('Countplot of Species')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(8, 6))
    sns.set_style("dark")
    sns.countplot(x='species', data=df,)
    plt.title('Countplot of Species')
    plt.xlabel('Species')
    plt.ylabel('Count')
    plt.tight_layout()
    st.pyplot()

    # Countplot of Sex
    st.subheader('')
    st.subheader('')
    st.subheader('')
    st.subheader('Countplot of Sex')
    plt.figure(figsize=(8, 6))
    sns.set_style("dark")
    sns.countplot(x='sex', data=df)
    plt.title('Countplot of Sex')
    plt.xlabel('Sex')
    plt.ylabel('Count')
    plt.tight_layout()
    st.pyplot()
    st.subheader('')
    st.subheader('')

    # Distribution of Numerical Variables
    st.subheader('Distribution of Numerical Variables')
    numerical_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    for col in numerical_cols:
        st.subheader('')
        st.write(f'Distribution of {col}')
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], bins=20, kde=True, edgecolor='black')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        st.pyplot()

    # Pairplot
    st.subheader('')
    st.subheader('')
    st.subheader('Pairplot of Numerical Variables')
    pairplot = sns.pairplot(df, hue='species')
    st.pyplot(pairplot.fig)


if __name__ == '__main__':
    main()
