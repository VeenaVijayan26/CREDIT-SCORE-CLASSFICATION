from flask import Flask, render_template, request
import pandas as pd
import pickle
import joblib  # Ensure joblib is imported
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load the saved encoder
encoder = joblib.load('encoder.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    data = {
        'Age': float(request.form['Age']),
        'Occupation': request.form['Occupation'],
        'Annual_Income': float(request.form['Annual_Income']),
        'Num_Bank_Accounts': float(request.form['Num_Bank_Accounts']),
        'Num_Credit_Card': float(request.form['Num_Credit_Card']),
        'Interest_Rate': float(request.form['Interest_Rate']),
        'Num_of_Loan': float(request.form['Num_of_Loan']),
        'Delay_from_due_date': float(request.form['Delay_from_due_date']),
        'Num_of_Delayed_Payment': float(request.form['Num_of_Delayed_Payment']),
        'Changed_Credit_Limit': float(request.form['Changed_Credit_Limit']),
        'Num_Credit_Inquiries': float(request.form['Num_Credit_Inquiries']),
        'Credit_Mix': request.form['Credit_Mix'],
        'Outstanding_Debt': float(request.form['Outstanding_Debt']),
        'Credit_Utilization_Ratio': float(request.form['Credit_Utilization_Ratio']),
        'Credit_History_Age': float(request.form['Credit_History_Age']),
        'Payment_of_Min_Amount': request.form['Payment_of_Min_Amount'],
        'Total_EMI_per_month': float(request.form['Total_EMI_per_month']),
        'Amount_invested_monthly': float(request.form['Amount_invested_monthly']),
        'Payment_Behaviour': request.form['Payment_Behaviour'],
        'Monthly_Balance': float(request.form['Monthly_Balance'])
    }
        # Convert the data to a DataFrame
    df = pd.DataFrame([data])

     # List of numeric columns to apply log transformation
    numeric_columns = [
        'Age', 'Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card',
        'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
        'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt',
        'Credit_Utilization_Ratio', 'Credit_History_Age', 'Total_EMI_per_month',
        'Amount_invested_monthly', 'Monthly_Balance'
    ]

    # Apply log1p transformation to numeric columns to handle any zeros
    df[numeric_columns] = df[numeric_columns].apply(np.log1p)

    # Transform the new data using the saved encoder
    encoded_new_data = encoder.transform(df)

    # Scale the encoded data
    data_scaled = scaler.transform(encoded_new_data)

    # Make predictions using the model
    prediction = model.predict(data_scaled)[0]

    # Map the prediction to the credit score
    if prediction == 0:
        credit_score = "Good "
    elif prediction == 1:
        credit_score = "Poor "
    else:
        credit_score = "Standard "
    
    return render_template('result.html', prediction_text=f'Your Credt score is: {credit_score}')

if __name__ == '__main__':
    app.run(debug=True)