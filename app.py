from flask import Flask, request, render_template
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model = joblib.load('best_model.pkl')

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = {
        'Application mode': request.form['application_mode'],
        'Displaced': request.form['displaced'],
        'Debtor': request.form['debtor'],
        'Tuition fees up to date': request.form['tuition_fees_up_to_date'],
        'Gender': request.form['gender'],
        'Scholarship holder': request.form['scholarship_holder'],
        'Age at enrollment': request.form['age_at_enrollment'],
        'Curricular units 1st sem (enrolled)': request.form['curricular_units_1st_sem_enrolled'],
        'Curricular units 1st sem (approved)': request.form['curricular_units_1st_sem_approved'],
        'Curricular units 1st sem (grade)': request.form['curricular_units_1st_sem_grade'],
        'Curricular units 2nd sem (enrolled)': request.form['curricular_units_2nd_sem_enrolled'],
        'Curricular units 2nd sem (approved)': request.form['curricular_units_2nd_sem_approved'],
        'Curricular units 2nd sem (grade)': request.form['curricular_units_2nd_sem_grade']
    }
    
    # Convert form data to DataFrame
    df = pd.DataFrame([data])

    # Predict using the loaded model
    prediction = model.predict(df)[0]

    # Map prediction to class names
    classes = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
    result = classes.get(prediction, 'Unknown')

    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)
