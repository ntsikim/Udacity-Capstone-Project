# Churn Prediction for Sparkify Users

## Table of Contents
- [Introduction](#introduction)
- [Project Definition](#project-definition)
  - [Project Overview](#project-overview)
  - [Problem Statement](#problem-statement)
  - [Metrics](#metrics)
- [Analysis](#analysis)
  - [Data Exploration](#data-exploration)
  - [Data Visualization](#data-visualization)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Implementation](#implementation)
  - [Refinement](#refinement)
- [Results](#results)
  - [Model Evaluation and Validation](#model-evaluation-and-validation)
  - [Justification](#justification)
- [Conclusion](#conclusion)
  - [Reflection](#reflection)
  - [Improvement](#improvement)
- [Deliverables](#deliverables)
  - [Write-up or Application](#write-up-or-application)
  - [GitHub Repository](#github-repository)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Introduction
This project aims to predict customer churn for Sparkify, a digital music service, using user activity data. By identifying patterns in user behavior that indicate a likelihood of churn, Sparkify can take proactive measures to retain users, thereby increasing overall customer satisfaction and reducing churn rates.

For more detailed information about the project, please refer to the [blog post](https://medium.com/@ntsikelelomyesi43/a7f7fd06bb30).

## Project Definition

### Project Overview
Customer churn is a critical problem for businesses in subscription-based industries. High churn rates can significantly impact revenue and growth. This project uses machine learning techniques to analyze user activity data and predict which users are likely to churn.

### Problem Statement
The objective is to build a machine learning model that accurately predicts whether a user will churn based on their activity patterns. The expected solution involves creating features from user data, training various models, and selecting the best-performing model based on evaluation metrics.

### Metrics
The primary metric used to evaluate model performance is the F1 score, which balances precision and recall. Given the class imbalance (fewer churned users), the F1 score is appropriate as it considers both false positives and false negatives.

## Analysis

### Data Exploration
The dataset includes user activities such as song plays, thumbs up/down interactions, friend additions, and playlist additions. Key features and statistics were calculated to understand user behavior. Abnormalities, such as missing values and invalid user IDs, were identified and addressed.

### Data Visualization
Visualizations were created to convey insights from the data exploration phase, including histograms of user interactions, box plots of session lengths, and bar charts comparing behaviors of churned vs. non-churned users.

## Methodology

### Data Preprocessing
Preprocessing steps included:
- Removing rows with missing or invalid user IDs.
- Creating a churn label based on 'Cancellation Confirmation' events.
- Extracting features like the number of songs played, total session length, and various user interactions.

### Implementation
Three models were implemented and evaluated:
- Logistic Regression
- Random Forest
- Gradient-Boosted Trees

Each model was trained and evaluated using cross-validation to ensure robustness. Hyperparameter tuning was performed to optimize model performance.

### Refinement
The refinement process involved iteratively improving feature selection and model parameters. The Random Forest model showed the best performance, and its hyperparameters were tuned using grid search.

## Results

### Model Evaluation and Validation
The Random Forest model achieved an F1 score of 0.7063 on the test set, indicating strong performance in predicting churn. The model was validated using cross-validation, and its parameters were fine-tuned for optimal performance.

### Justification
The Random Forest model was chosen due to its superior F1 score, which reflects its ability to effectively balance precision and recall. The model's performance indicates that user engagement metrics are crucial for predicting churn.

## Conclusion

### Reflection
This project successfully built and validated a churn prediction model using user activity data. The main challenges included handling class imbalance and managing large datasets. Key insights were gained into the impact of user interactions on churn prediction.

### Improvement
Potential improvements include:
- Exploring additional features like time since last login or user demographics.
- Combining multiple models to create an ensemble for potentially better performance.
- Implementing the model in a real-time system for actionable insights.

## Deliverables

### Write-up or Application
The project report is well-organized and aimed at a technical audience. It includes all relevant sections, such as project overview, methodology, results, and conclusion. The code follows PEP8 guidelines and best practices, ensuring readability and maintainability.

### GitHub Repository

- `README.md`: Overview of the project, methodology, results, and instructions for reproducing the analysis.
- `Sparkify_Churn_Prediction.ipynb`: The complete Jupyter Notebook with step-by-step analysis and visualizations.
- `Sparkify_Churn_Prediction.py`: The cleaned and well-documented Python script for the project.
- `requirements.txt`: List of dependencies required to run the project.
- `images/`: Directory containing data visualization images used in the analysis.

## Usage

1. **Setup Environment**:
   - Create a virtual environment and activate it:
     ```bash
     python3 -m venv env
     source env/bin/activate  # On Windows, use `env\Scripts\activate`
     ```

2. **Install Dependencies**:
   - Install the required libraries:
     ```bash
     pip install -r requirements.txt
     ```

3. **Jupyter Notebook**:
   - Open `Sparkify_Churn_Prediction.ipynb` in Jupyter Notebook or Jupyter Lab to explore the interactive analysis.

4. **Python Script**:
   - Run `Sparkify_Churn_Prediction.py` in your Python environment to execute the complete analysis.

## Acknowledgments
This project is part of the Data Scientist Nanodegree program. Special thanks to the instructors and mentors for their guidance and support.
