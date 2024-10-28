

# Health Prediction Web Application

This project is a simple web application for predicting the likelihood of heart disease based on age, gender, blood pressure, and cholesterol levels. It uses a machine learning model built with Python and Flask as the backend server and a basic HTML frontend for the user interface.

## Project Structure

```
health_prediction_app/
├── templates/
│   └── index.html               # The main HTML file for the web interface
├── health_model.joblib          # The trained machine learning model
├── heart_dataset.csv            # The dataset used for training the model
└── index.py                     # Main Python script to run the Flask app
```

- `templates/index.html`: Contains the form and UI for users to input health metrics and receive predictions.
- `health_model.joblib`: The saved machine learning model file, used by `index.py` to make predictions.
- `heart_dataset.csv`: Dataset containing health data for model training.
- `index.py`: Main Python script that runs the Flask app, loads the model, and handles prediction requests.

## Installation and Setup

### Prerequisites

- Python 3.7 or higher
- Flask
- Flask-CORS
- Scikit-learn
- Joblib
- Pandas
- Numpy

### Setup Instructions

1. **Clone the repository** (or download the files).
   
2. **Install required packages**. You can install all dependencies by running:

    ```
    pip install flask flask-cors scikit-learn joblib pandas numpy
    ```

3. **Set up the dataset**:
    - Ensure that `heart_dataset.csv` is in the root folder of the project.
    - This dataset should contain the columns `age`, `gender`, `blood_pressure`, `cholesterol_levels`, and the target variable column `target_variable` (used for training).

4. **Train the model** (Optional):
    - If you want to retrain the model, you can load the dataset, preprocess it, and save the model again as `health_model.joblib`.
    - `index.py` already includes code for training and saving the model, which you can uncomment to retrain.

5. **Start the Flask Application**:
    - In the project directory, run:

    ```
    python index.py
    ```

6. **Access the Application**:
    - Open a web browser and go to `http://127.0.0.1:5000` to view the application.

## How the Model Works

1. **Data Preprocessing**:
    - The dataset, `heart_dataset.csv`, contains health-related data that is preprocessed to ensure no missing values and encoded where necessary (e.g., encoding gender as 0 for Male and 1 for Female).
  
2. **Model Training**:
    - A linear regression model is trained using features: `age`, `gender`, `blood_pressure`, and `cholesterol_levels`, with the target variable being the likelihood of heart disease (`target_variable`).

3. **Prediction**:
    - When a user enters values into the form on `index.html` and clicks "Get Prediction," these values are sent as a POST request to the Flask server.
    - The model loaded in `index.py` processes the input data and returns a prediction as a probability percentage, indicating the likelihood of heart disease.

## Endpoints

- `GET /`: Serves the main HTML form (`index.html`).
- `POST /predict`: Accepts JSON data containing `age`, `gender`, `blood_pressure`, and `cholesterol_levels` and returns a prediction as a JSON response.

## Example Usage

- Enter an age, gender, blood pressure, and cholesterol level in the form.
- Click "Get Prediction."
- The app will display a percentage likelihood of heart disease.

## Troubleshooting

- **Model not loading**: Ensure that `health_model.joblib` exists and is in the correct directory.
- **Data file not found**: Make sure `heart_dataset.csv` is available and that the file path in `index.py` is correct.
- **Module not found errors**: Run `pip install` for any missing dependencies.

