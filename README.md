# Loan Approval Prediction API

This project is a Flask-based API that predicts loan approval status based on client financial data. It allows users to submit financial data via a web form or programmatically via a POST request, and returns whether a loan will be approved or rejected. The application is packaged using Docker for easy deployment.


## Directory Structure
## Directory Structure

```plaintext
C:.
│   app.py
│   app_s.py
│   Dockerfile
│   LICENSE
│   loan.ipynb
│   loan_approval_dataset_updated.csv
│   loan_approval_model.pkl
│   rb_scaler.pkl
│   README.md
│   requirements.txt
│   scaler.pkl
│
├───.ipynb_checkpoints
│       loan-checkpoint.ipynb
│       Loan_Eligibility_Prediction_using_Gradient_Boosting_Classifier_prediction-checkpoint.ipynb
│
└───templates
        index.html
        result.html





## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Running Locally](#running-locally)
  - [Running with Docker](#running-with-docker)
- [API Endpoints](#api-endpoints)
- [Logging and Error Handling](#logging-and-error-handling)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview
This API takes client financial data such as annual income, loan amount, credit score, and asset values as inputs, and predicts whether the loan will be approved or rejected based on a pre-trained machine learning model (XGBoost).

## Requirements
- **Python** 3.8 or higher
- **Flask** 3.0.3
- **XGBoost** 1.7.6
- **Pandas** 1.3.5
- **Scikit-learn** 1.3.0
- **Joblib** 1.3.1
- **Docker**

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/loan-approval-api.git
cd loan-approval-api


