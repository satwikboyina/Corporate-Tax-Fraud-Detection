from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the scaler and model pipeline
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Define a route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the HTML form
    form_data = request.form.to_dict()

    # Remove commas and percentage signs from numeric strings
    for key, value in form_data.items():
        if ',' in value:
            value = value.replace(',', '')
        if '%' in value:
            value = value.replace('%', '')
        if '$' in value:
            value = value.replace('$', '')    
        form_data[key] = value

    # Create a DataFrame from the user input
    user_df = pd.DataFrame([form_data])

    # Ensure the correct order of columns
    user_df = user_df[['Corporate Income Tax Rate', 'Net Sales', 'Operating Expenses', 'Other Income',
                       'Gross Profit Margin', 'Operating Cash Flow', 'Current Assets', 'Current Liabilities',
                       'Effective Tax Rate', 'Inventory Turnover', 'Return on Assets (ROA)', 'Number of Employees']]

    # Convert user input to numeric format
    user_df = user_df.astype(float)

    # Scale the user input data
    user_scaled = scaler.transform(user_df)

    # Predict with the model pipeline
    prediction = pipeline.predict(user_scaled)

    # Interpret prediction (assuming 1 means 'paid' and 0 means 'not paid')
    result = "paid" if prediction[0] == 1 else "not paid"

    # Render the result
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
