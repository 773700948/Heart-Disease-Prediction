# Heart-Disease-Prediction
## Code Overview
### Importing Dependencies:

The code begins by importing necessary libraries such as numpy, pandas, train_test_split from sklearn.model_selection, LogisticRegression from sklearn.linear_model, and accuracy_score from sklearn.metrics.

### Data Collection and Processing:

The dataset is loaded into a pandas DataFrame using pd.read_csv.

Basic data exploration is performed, including displaying the first and last few rows of the dataset, checking the shape of the dataset, and obtaining statistical measures using describe().

The distribution of the target variable (target) is analyzed using value_counts().

### Data Preprocessing:

The dataset is split into features (X) and labels (Y).

The features are standardized using StandardScaler to ensure that all features have a mean of 0 and a standard deviation of 1.

### Train-Test Split:

The dataset is split into training and testing sets using train_test_split with a test size of 20% and stratified sampling based on the target variable.

### Model Training:

A Logistic Regression model is initialized and trained on the training data.

### Model Evaluation:

The model's accuracy is evaluated on both the training and test datasets using accuracy_score.

### Making Predictions:

A predictive system is implemented where a new data point is passed to the trained model for prediction. The model predicts whether the person has heart disease or not.

## Dataset Overview
The dataset used in this project is the Heart Disease Dataset, which contains the following features:

age: Age of the patient.

sex: Gender of the patient (1 = male, 0 = female).

cp: Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic).

trestbps: Resting blood pressure (in mm Hg).

chol: Serum cholesterol level (in mg/dl).

fbs: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false).

restecg: Resting electrocardiographic results (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy).

thalach: Maximum heart rate achieved.

exang: Exercise-induced angina (1 = yes, 0 = no).

oldpeak: ST depression induced by exercise relative to rest.

slope: Slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping).

ca: Number of major vessels (0-3) colored by fluoroscopy.

thal: Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect).

target: Target variable (0 = no heart disease, 1 = heart disease).

###  Key Points
Data Preprocessing: The dataset is checked for missing values, and no missing values are found. The features are standardized to ensure that the Logistic Regression model performs well.

Model Choice: Logistic Regression is chosen for its effectiveness in binary classification tasks, especially when the data is not too large.

Evaluation: The model's performance is evaluated using accuracy, which is a common metric for classification tasks.

Example Prediction
The code includes an example where a new data point (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2) is used to predict whether the person has heart disease. The model predicts that the person does not have heart disease.

# Conclusion
This project demonstrates a typical workflow for a binary classification problem using machine learning. It covers data loading, exploration, preprocessing, model training, evaluation, and prediction. The use of Logistic Regression is a common approach for such tasks, and the code provides a clear example of how to implement this in Python.

## Dataset Summary
Number of Rows: 303

Number of Columns: 14

Target Variable: target (0 = no heart disease, 1 = heart disease)

Features: 13 features related to patient health metrics.

## Model Performance
Training Accuracy: ~85.12%

Test Accuracy: ~81.97%

The model performs reasonably well on both the training and test datasets, indicating that it generalizes well to unseen data.

## Example Prediction Output
For the input data (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2), the model predicts:

The Person does not have a Heart Disease
This indicates that the model predicts the person does not have heart disease based on the provided features.

## Next Steps
Feature Engineering: Explore additional feature engineering techniques to improve model performance.

Hyperparameter Tuning: Experiment with different hyperparameters for the Logistic Regression model.

Other Models: Try other classification models such as Random Forest, SVM, or Neural Networks to compare performance.

