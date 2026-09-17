# 🤟 Hand Sign Recognition using Classical Machine Learning

A Computer Vision and Machine Learning project for recognizing hand signs and predicting the corresponding alphabet/class from images.

The project uses **MediaPipe** to detect hands and extract hand landmarks, then applies **Classical Machine Learning** algorithms to classify the extracted features.

Multiple machine learning models are trained, evaluated, and compared to identify the best-performing model for hand sign recognition.

---

## 📌 Project Overview

The main goal of this project is to build a Hand Sign Recognition system capable of predicting the class represented by a hand gesture.

Instead of training machine learning models directly on raw images, the project first detects the hand and extracts its landmark coordinates using MediaPipe.

The extracted landmark features are then converted into a structured CSV dataset and used to train Classical Machine Learning models.

### Main Pipeline

```text
Hand Sign Images
       │
       ▼
Hand Detection
       │
       ▼
Hand Landmark Extraction
       │
       ▼
Feature Engineering
       │
       ▼
dataset.csv
       │
       ▼
Data Preprocessing
       │
       ▼
Train / Test Split
       │
       ▼
Classical Machine Learning
       │
       ├───────────────┐
       ▼               ▼
Random Forest        SVM
       │               │
       └───────┬───────┘
               │
               ▼
              KNN
               │
               ▼
       Model Evaluation
               │
               ▼
       Model Comparison
               │
               ▼
          Best Model
               │
               ▼
     best_sign_model.pkl
               │
               ▼
           Prediction



🎯 Project Objectives
Detect hands from input images.
Extract hand landmark coordinates.
Convert hand landmarks into numerical features.
Create a structured CSV dataset.
Preprocess the extracted features.
Train multiple Classical Machine Learning models.
Compare different classification algorithms.
Evaluate model performance using multiple metrics.
Select the best-performing model based on evaluation results.
Save the trained model for future predictions.


🧠 Computer Vision Approach
The project uses hand landmarks instead of raw image pixels as the main input features.
MediaPipe detects the hand in each image and extracts landmark points representing important locations of the hand.
The coordinates of these landmarks are converted into numerical features.
These features are then stored in a CSV file together with the corresponding hand-sign label.
Example Feature Structure
Feature Description
x1, y1
Landmark 1 coordinates
x2, y2
Landmark 2 coordinates
x3, y3
Landmark 2 coordinates
...
...
xN, yN
Landmark N coordinates
label
Hand sign class
This feature-based approach allows Classical Machine Learning algorithms to work with compact numerical representations of hand gestures.


📂 Dataset
The original dataset consists of images representing different hand signs/classes.
Dataset Source
The dataset used in this project is available here:
Download Hand Sign Dataset⁠�
The original images are processed using the Computer Vision pipeline to extract hand landmarks and generate the final CSV dataset used for Machine Learning.



🔄 Data Processing Pipeline
1. Image Collection
The project starts with a collection of hand-sign images representing different classes.
2. Hand Detection
MediaPipe is used to detect the hand present in each image.
Images where a valid hand cannot be detected can be handled or excluded during preprocessing.
3. Landmark Extraction
After detecting the hand, MediaPipe extracts the hand landmark coordinates.
Each landmark contains coordinate information such as: X,Y.


4. Feature Generation
The extracted landmark coordinates are converted into numerical features suitable for Classical Machine Learning algorithms.
5. CSV Dataset Creation
The extracted features and their corresponding labels are stored in dataset.csv.
The CSV file provides a structured representation of the original image dataset.

6. Data Cleaning The generated dataset is checked for:
Missing values
Invalid samples
Duplicate records
Incorrect labels
Incomplete landmark features.

7. Data Splitting The dataset is divided into training and testing sets.
A stratified split is used to preserve the class distribution between the training and testing datasets.


🤖 Machine Learning Models
The project compares several Classical Machine Learning classification algorithms.
🌲 Random Forest
Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions.
It is used as one of the main classification models for the hand-sign recognition task.
📐 Support Vector Machine
Support Vector Machine (SVM) is used with an RBF kernel to learn nonlinear decision boundaries between different hand-sign classes.
👥 K-Nearest Neighbors
K-Nearest Neighbors (KNN) classifies a new sample based on the labels of its nearest training samples.


⚙️ Model Training
Each model is trained using the extracted hand landmark features.
The general training workflow is:
dataset.csv
     │
     ▼
Feature / Label Separation
     │
     ▼
Train / Test Split
     │
     ▼
Feature Preprocessing
     │
     ▼
Model Training
     │
     ▼
Prediction
     │
     ▼
Evaluation

The same evaluation process is applied to the different models to make the comparison consistent.


📊 Model Evaluation
The models are evaluated using several classification metrics:
Accuracy
Precision
Recall
Macro F1-Score
Confusion Matrix
Training Time
Prediction Time
Accuracy
Accuracy measures the percentage of correctly classified samples.
Precision
Precision measures how many samples predicted as a specific class actually belong to that class.
Recall
Recall measures how many samples of a specific class were correctly detected.
Macro F1-Score
Macro F1 calculates the F1-score independently for each class and then takes the average.
This is useful for multi-class classification because each class receives equal importance.
Confusion Matrix
The confusion matrix shows how predictions are distributed across the different hand-sign classes and helps identify classes that are frequently confused with each other.


🏆 Model Comparison
The trained models are compared using their evaluation metrics.


Random Forest:
Train Accuracy: 100.00%
Test Accuracy: 95.65%
Train F1_Score: 1.0000
Test F1_Score: 0.9500
Time (s): 1.47

=============================

SVM (RBF)
Train Accuracy : 96.26%
Test Accuracy :96.03%
Train F1_Score: 0.9570
Test F1_Score: 0.9523 
Time (s): 0.09

                      
===============================

KNN:
Train Accuracy : 95.27%
Test Accuracy :92.82%
Train F1_Score: 0.9449 
Test F1_Score: 0.9190 
Time (s): 0.01

The final model is selected based on the evaluation results obtained from the test dataset.
The values above should be replaced with the actual results generated by the project.



## 📊 Results & Visualizations

The project includes detailed visualizations and model evaluation in the Jupyter Notebook, including:

- Hand landmark visualization
- Class distribution
- Model performance comparison
- Confusion Matrix
- Classification results

See the notebook for the complete analysis and visualizations.



💾 Model Persistence
After model evaluation, the selected model is saved using Joblib.
Example:
best_sign_model.pkl
Saving the trained model makes it possible to reuse it later without retraining the entire model.
Load the Saved Model
import joblib
model = joblib.load("best_sign_model.pkl")


🛠️ Technologies
Python
OpenCV
MediaPipe
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
Joblib

📁 Project Structure
computer_cv/
│
├── dataset.csv
├── best_sign_model.pkl
├── extract_landmarks.py
├── train_evaluate.py
├── realtime.py
├── visualization.ipynb
├── requirements.txt
└── README.md


🚀 Future Improvements
Real-time webcam recognition.
More hand-sign classes.
Hyperparameter optimization.
Web application deployment.

👨‍💻 Author
Mohamed Said Abdelaziz
Computer Science Student
