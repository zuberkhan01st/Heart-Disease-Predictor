import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    st.title('Heart Disease Checkup')
    
    # Load the heart disease dataset
    data = pd.read_csv('heart.csv')

    # Separate features (x) and target (y)
    columns_to_drop = ['target', 'slope', 'ca', 'thal']
    x = data.drop(columns_to_drop, axis=1)
    y = data['target']

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Function to get user input
    def user_report():
        age = st.slider('Patients age', 20, 90, 40)
        sex = st.radio('Patients sex', ['Male', 'Female'])
        cp = st.slider('Chest Pain Type', 0, 3, 0)
        trestbps = st.slider('Resting Blood Pressure', 90, 200, 120)
        chol = st.slider('Serum Cholesterol', 100, 600, 200)
        fbs = st.radio('Fasting Blood Sugar', ['< 120 mg/dl', '> 120 mg/dl'])
        restecg = st.slider('Resting ECG Results', 0, 2, 0)
        thalach = st.slider('Maximum Heart Rate Achieved', 60, 220, 150)
        exang = st.radio('Exercise Induced Angina', ['Yes', 'No'])
        oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 6.2, 0.0)

        user_report = {
            'age': age,
            'sex': 1 if sex == 'Male' else 0,  # Encoding Male as 1 and Female as 0
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': 1 if fbs == '> 120 mg/dl' else 0,  # Encoding > 120 mg/dl as 1 and < 120 mg/dl as 0
            'restecg': restecg,
            'thalach': thalach,
            'exang': 1 if exang == 'Yes' else 0,  # Encoding Yes as 1 and No as 0
            'oldpeak': oldpeak,
        }
        report_data = pd.DataFrame(user_report, index=[0])
        return report_data

    # Get user input
    user_data = user_report()

    # Train Random Forest classifier
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    # Predict user's heart disease status
    st.subheader('Your Report')
    user_result = rf.predict(user_data)
    if user_result[0] == 0:
        st.write('You are safe from heart disease.')
    else:
        st.write('You are facing heart disease.')

if __name__ == "__main__":
    main()
