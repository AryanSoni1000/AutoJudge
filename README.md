AutoJudge:Programming Problem Difficulty Predictor

Overview

AutoJudge: This work descirbes a purely end-to-end Machine Learning + NLP based system that predicts the difficulty of programming problems using only their textual descriptions.

The system does the following two operations:

1. Classification: Predicts whether the problem is Easy / Medium / Hard.
2. Regression: Outputs a continuous difficulty score.

A simple web interface using Streamlit allows users to paste a problem description and get predictions immediately.


Dataset Used

These are some problems from online sites that were collected on programming tasks.

Each sample of the data contains the following fields:

title
description
input_description
output_description
problem_class (Easy / Medium / Hard)
problem_score (numerical difficulty score)

Only textual information is used. Code, constraints, and metadata have no bearing.


Data Preprocessing

Loaded dataset from a .jsonl file
Handled missing values by filling them with empty strings
Combined all text fields into a single column full_text for NLP processing


Feature Engineering

TF-IDF Vectorization (top 5000 features) to represent the text in numerical form

Other additional manually crafted features:

Text length
Competitive programming keyword frequency (such as dp, graph, tree, recursion, etc.)


Models Used

1. Classification Model

Model: Logistic Regression
Activity: Predict difficulty class (Easy / Medium / Hard)

2. Regression Model

Model: Random Forest Regressor
Activity: Predict numerical value of difficulty


Evaluation Metrics

Classification

Accuracy: 53%
Confusion Matrix used for class-wise performance analysis

Since this is a three class classification problem with labeled data, the baseline accuracy of random guessing is about 33%.

The result achieved is remarkably better than the baseline.

Regression

Mean Absolute Error (MAE): 1.69
Root Mean Squared Error (RMSE): 2.03

These findings suggest that the model is capable of making a good estimate of the difficulty score just from the text.


Web Interface

It is developed using Streamlit for web application generation.

Attributes:

Text input fields for:

Problem Title
Problem Description
Input Description
Output Description

A “Predict Difficulty” button

Displays:

Predicted difficulty class
Predicted numerical difficulty score

The application runs locally and loads pre-trained models for inference purposes.


User Flow:

1. The user provides the problem title, description, input details, and output details.
2. Upon clicking the button “Predict Difficulty”, the text is analyzed, turning the input text into numbers.
3. The trained models for classification and regression are then used for predictions.
4. The predicted class of difficulty and difficulty score value will appear on the GUI.


The screenshots for web interface and sample predictions are provided with this report


Conclusion:

In this project, it is shown that significant patterns of difficulty can be obtained from the description of the problem in programming tasks in classical NLP and machine learning.
Despite being based only on text data and human judgements of programming task difficulty, the model is shown capable of predicting both the class and score with reasonable performance. The model’s performance was measured in terms of its ability to approximate the human judgements of The project also emphasizes the relevance of effective feature engineering and analysis within actual ML tasks.


Link:
Here is the demo video link :
https://drive.google.com/file/d/1FXM557jdhEPgr9OhxmJlVn3WPCQrF5k5/view?usp=sharing

Details:
Name- Aryan Soni
Branch - Chemical Engineering
Year-2nd
Enrollment number - 24112025
Institute- IIT Roorkee
