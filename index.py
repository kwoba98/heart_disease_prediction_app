import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Load the dataset
df = pd.read_csv("C:\\AI programming with PYTHON\\health_prediction_app\\heart_dataset.csv")


# Select relevant features and target variable
X = df[['age', 'gender', 'blood_pressure', 'cholesterol_levels']]
y = df['target_variable']  # Ensure this is a continuous variable for linear regression

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a file
model_filename = 'health_model.joblib'
joblib.dump(model, model_filename)

# Load the model (useful for reloading without retraining)
model = joblib.load(model_filename)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation using Linear Regression:")
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Create a Flask application
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')  # Ensure 'index.html' exists in a 'templates' folder

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        age = int(data['age'])
        gender = int(data['gender'])
        blood_pressure = int(data['blood_pressure'])
        cholesterol_levels = int(data['cholesterol_levels'])

        # Prepare the input for the model
        input_data = np.array([[age, gender, blood_pressure, cholesterol_levels]])
        
        # Make a prediction and return as a float
        prediction = model.predict(input_data)
        prediction_value = float(prediction[0])  # Convert to a Python float

        return jsonify({'prediction': prediction_value})
    
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
