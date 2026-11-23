# Student Exam Score Predictor

A machine learning-powered web application that predicts student exam scores based on academic, behavioral, and lifestyle factors. Built with Python, Scikit-learn, and Streamlit, the application provides predictions along with personalized insights and visualizations to help understand factors associated with student performance.

## Overview

Student academic performance is influenced by multiple factors, including study habits, attendance, sleep quality, learning resources, and academic environment.

This project uses a machine learning model to estimate a student's exam score based on these factors. It combines a trained prediction model with an interactive web interface, allowing users to explore how different student characteristics relate to predicted academic outcomes.

## Features

- **Exam Score Prediction:** Predict student exam scores using multiple academic and lifestyle inputs.
- **Interactive Web Interface:** Enter student information through a user-friendly Streamlit application.
- **Student Insights:** Generate rule-based insights regarding study hours, attendance, sleep quality, and internet accessibility.
- **Data Visualization:** Visualize key student parameters through graphical comparisons.
- **Machine Learning Pipeline:** Process input features using the trained model and preprocessing techniques.
- **Model Persistence:** Serialize and load the trained model using Joblib.
- **Cloud Deployment:** Deploy the application through Streamlit Community Cloud.
- **Dynamic Model Loading:** Download the model from Google Drive when required, avoiding the need to store a large model file directly in the repository.

## Input Features

The model uses the following student-related features:

| Feature | Description |
|---|---|
| Age | Student's age |
| Gender | Gender category |
| Course | Academic course or program |
| Study Hours | Daily study duration |
| Class Attendance | Attendance percentage |
| Internet Access | Availability of internet access |
| Sleep Hours | Daily sleep duration |
| Sleep Quality | Self-reported sleep quality |
| Study Method | Preferred learning or study method |
| Facility Rating | Rating of available academic facilities |
| Exam Difficulty | Perceived examination difficulty |

## How It Works

The application follows the workflow below:

1. The user enters student-related information through the Streamlit interface.
2. Input features are organized into a structured DataFrame.
3. The trained machine learning model processes the input data.
4. The model generates a predicted exam score.
5. The application displays the prediction.
6. Rule-based insights are generated based on selected student attributes.
7. A visualization presents selected student parameters for comparison.

## Machine Learning

The project uses a supervised machine learning regression approach to predict numerical exam scores.

### Model Pipeline

- Data preprocessing
- Categorical feature encoding using One-Hot Encoding
- Model training and evaluation
- Model serialization using Joblib
- Model loading for inference
- Prediction using new input data

The trained model is loaded into the application and used to generate predictions without requiring model training every time the application starts.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and input preparation |
| Scikit-learn | Machine learning and preprocessing |
| Streamlit | Interactive web application |
| Matplotlib | Data visualization |
| Joblib | Model serialization and loading |
| gdown | Downloading the model from Google Drive |

## Project Structure

```text
student-score-predictor/
│
├── app.py
├── eda-and-ml-model.ipynb
├── requirements.txt
├── .gitignore
│
└── .devcontainer/
```

| File | Description |
|---|---|
| `app.py` | Main Streamlit application containing the prediction interface, model loading, insights, and visualizations |
| `eda-and-ml-model.ipynb` | Notebook for exploratory data analysis and machine learning experimentation |
| `requirements.txt` | Python dependencies required to run the application |
| `.gitignore` | Specifies files and directories excluded from version control |
| `.devcontainer/` | Development environment configuration |

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Noshaba-Fatima/student-score-predictor.git
```

Navigate into the project directory:

```bash
cd student-score-predictor
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## Model Storage and Deployment

The trained model is hosted externally on Google Drive because of its file size.

The application uses `gdown` to download the model when it is not already available locally. Streamlit's resource caching is used to avoid repeatedly loading the model during application execution.

This approach keeps the GitHub repository lightweight while allowing the deployed application to access the trained model.

## Limitations

- Predictions depend on the quality, distribution, and representativeness of the training data.
- The predicted score is an estimate and should not be treated as a guaranteed academic outcome.
- Rule-based insights are based on predefined thresholds and are not generated by the machine learning model.
- Relationships learned by the model do not necessarily imply causation.
- Prediction performance may vary for students whose characteristics differ significantly from the training dataset.

## Future Improvements

- Add model evaluation metrics such as MAE, RMSE, and R² to the application.
- Incorporate feature importance and explainability techniques.
- Improve the visualization dashboard with additional analytical charts.
- Add prediction history and downloadable reports.
- Improve input validation and error handling.
- Containerize the application using Docker.
- Integrate automated model retraining and deployment workflows.

## Author

**Noshaba Fatima**

GitHub: [Noshaba-Fatima](https://github.com/Noshaba-Fatima)

## Project Repository

[Student Score Predictor](https://github.com/Noshaba-Fatima/student-score-predictor)
