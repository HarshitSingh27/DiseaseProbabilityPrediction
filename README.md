# Disease Probability Prediction through Multiple Feature Correlation

## Project Overview

Developed between January 2023 and June 2023, this project introduces a novel approach to medical outcome prediction using Python and Bayesian networks. By leveraging patient data, we've created a web app that offers personalized disease risk predictions. Our research, presented at ICRAET-2023[https://drive.google.com/file/d/1mfu1AmHRJNPkHlLN1f-bDdFSW5gMHmzg/view?usp=sharing], underscores the potential of predictive analysis in revolutionizing diagnostics, providing healthcare professionals with an invaluable tool for improving patient care.

## Requirements

To run this project, you will need the following packages installed:

- streamlit==0.80.0
- pandas==1.2.4
- numpy==1.19.5
- scikit-learn==0.24.1

## How to Run

To start the web app, navigate to the directory where the file is downloaded and execute the following command in your terminal:
- streamlit run app.py

## Features

Our application processes various patient features, such as age, gender, symptoms, and medical history, to generate a list of potential diseases along with their corresponding probability scores. It identifies the highest correlation among multiple indicators, facilitating informed decision-making and personalized treatment plans.

![Demo Image](demo/demo.gif)

## Dataset

The project utilizes a comprehensive dataset, sdsp_patients.xlsx, that includes patient demographics, medical history, and outcomes. This dataset was instrumental in training our machine learning models to accurately predict disease probabilities.

## Machine Learning Model

We employed the Random Forest Classifier algorithm due to its robustness and ability to handle complex datasets. Through extensive preprocessing and feature selection, we optimized our model to deliver accurate predictions across four major diseases: Type-2 Diabetes, Coronary Heart Disease, Chronic Obstructive Pulmonary Disease (COPD), and Asthma.

## User Interface

The application boasts an intuitive web interface, developed using Flask, that allows users to easily input patient data and receive risk predictions. The sidebar facilitates seamless navigation and data entry, enhancing the user experience.

## Contributions and Future Work

This project lays the groundwork for future advancements in the field of medical diagnostics. We invite the community to contribute to our codebase, propose new features, or extend the model's capabilities to cover more diseases and conditions.

## Contact

For support or inquiries, please contact us at (hs58@illinois.edu)
