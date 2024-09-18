# Loan Approval Prediction API

This project is a Flask-based API that predicts loan approval status based on client financial data. It allows users to submit financial data via a web form or programmatically via a POST request, and returns whether a loan will be approved or rejected. The application is packaged using Docker for easy deployment.

## Directory Structure
loan-approval-api/
│
├── app.py               # Main Flask application
├── loan_approval_model.pkl  # Pre-trained XGBoost model file
├── rb_scaler.pkl        # Scaler used for feature scaling
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker build configuration
├── templates/           # HTML templates for Flask (index.html and result.html)
│   ├── index.html       # Home page with form for data entry
│   └── result.html      # Result page showing prediction
├── static/              # Static assets (CSS, JS, etc.)
├── README.md            # Project documentation (this file)
└── app.log              # Log file for recording application events





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


