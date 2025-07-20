import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib
import streamlit as st
import matplotlib.pyplot as plt

def generate_synthetic_data():
    """Generate a synthetic dataset if Salary.csv is not found."""
    np.random.seed(42)
    data = pd.DataFrame({
        'Age': np.random.randint(18, 70, 100),
        'Education Level': np.random.choice(['High School', 'Bachelor\'s', 'Master\'s', 'PhD'], 100),
        'Job Title': np.random.choice(['Software Engineer', 'Data Analyst', 'Manager', 'Sales Representative'], 100),
        'Years of Experience': np.random.randint(0, 40, 100),
        'Salary': np.random.uniform(30000, 120000, 100)
    })
    return data

def train_model():

    # Try loading the Kaggle dataset
    try:
        data = pd.read_csv('Salary.csv')
        print("Loaded Kaggle dataset: Salary.csv")
    except FileNotFoundError:
        st.warning("Salary.csv not found. Generating synthetic dataset instead.")
        data = generate_synthetic_data()

    # Select relevant features
    data = data[['Age', 'Education Level', 'Job Title', 'Years of Experience', 'Salary']]

    # Handle missing values
    data = data.dropna()

    # Preprocess data
    le_education = LabelEncoder()
    le_job = LabelEncoder()

    data['Education Level'] = le_education.fit_transform(data['Education Level'])
    data['Job Title'] = le_job.fit_transform(data['Job Title'])

    # Save label encoders
    joblib.dump(le_education, 'le_education.pkl')
    joblib.dump(le_job, 'le_job.pkl')

    # Define features and target
    X = data[['Age', 'Education Level', 'Job Title', 'Years of Experience']]
    y = data['Salary']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Initialize regression models
    models = {
        'DecisionTree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    }

    # Train and evaluate models
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        results[name] = rmse
        print(f'{name} RMSE: {rmse:.2f} USD')

    # Save the best model (lowest RMSE)
    best_model_name = min(results, key=results.get)
    best_model = models[best_model_name]
    joblib.dump(best_model, 'best_salary_model.pkl')
    print(f'\nBest Model: {best_model_name} with RMSE {results[best_model_name]:.2f} USD')
    print('Saved best model as best_salary_model.pkl')

    # Plot model performance
    plt.figure(figsize=(8, 5))
    plt.bar(results.keys(), results.values(), color='lightblue')
    plt.title('Model Performance Comparison (RMSE)')
    plt.ylabel('Root Mean Squared Error (USD)')
    plt.grid(True, axis='y')
    plt.savefig('model_performance.png')
    plt.close()

def run_streamlit_app():
    # Set page config as the first Streamlit command
    st.set_page_config(page_title='Salary Predictor', page_icon='💰', layout='centered')

    # Inject custom CSS for compact layout and fixed footer
    st.markdown("""
        <style>
        /* Reduce overall padding and margins */
        .main .block-container {
            max-width: 800px;
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        /* Style for the footer */
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: black;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
            z-index: 100;
        }
        /* Reduce input element spacing */
        .stSlider > div, .stSelectbox > div {
            margin-bottom: 0.5rem;
        }
        /* Compact text */
        h1, h2, p {
            margin: 0.3rem 0;
            font-size: 1.5rem; /* Smaller title */
        }
        h2 {
            font-size: 1.2rem; /* Smaller header */
        }
        .stButton > button {
            margin-top: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Load model and encoders
    try:
        model = joblib.load('best_salary_model.pkl')
        le_education = joblib.load('le_education.pkl')
        le_job = joblib.load('le_job.pkl')
    except FileNotFoundError:
        st.error("Model or encoder files not found. Please run the training process first.")
        return

    # Streamlit app content
    st.title('💰 Salary Predictor')
    st.markdown('Predict annual salary in USD.')

    # Input form using columns for compact layout
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider('Age', 18, 70, 30, step=1)
        education = st.selectbox('Education', ['High School', 'Bachelor\'s', 'Master\'s', 'PhD'], key='education')
    with col2:
        job = st.selectbox('Job Title', ['Software Engineer', 'Data Analyst', 'Manager', 'Sales Representative'], key='job')
        experience = st.slider('Experience (Years)', 0, 40, 5, step=1)

    # Prepare input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Education Level': [le_education.transform([education])[0]],
        'Job Title': [le_job.transform([job])[0]],
        'Years of Experience': [experience]
    })

    # Prediction
    if st.button('Predict Salary'):
        prediction = model.predict(input_data)[0]
        st.success(f'Predicted Salary: ${prediction:,.2f} USD')
        if prediction > 50000:
            st.balloons()
            st.markdown('🎉 High earner!')
        else:
            st.markdown(' Moderate salary.')

    # Footer with creator info, LinkedIn, and GitHub
    st.markdown("""
        <div class='footer'>
            <p><strong>Made by PRANAV CHAWLA</strong></p>
            <p>3rd Year B.E. @MBM University, Jodhpur</p>
            <p>
                <a href='https://www.linkedin.com/in/pranav-chawla-a95b78290/' target='_blank'>
                    <img src='https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png' alt='LinkedIn' width='25' style='vertical-align: middle; margin-right: 8px;'>
                    LinkedIn
                </a>
                  
                <a href='https://github.com/pranav5458' target='_blank'>
                    <img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' alt='GitHub' width='25' style='vertical-align: middle; margin-right: 8px;'>
                    GitHub
                </a>
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    train_model()
    run_streamlit_app()