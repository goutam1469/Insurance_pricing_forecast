# Insurance_pricing_forecast

The aim of this project is to build and evaluate linear and xgboost regression models and determine the healthcare charges of each customer. This analysis will help the insurance firm to strategize a premium plan that will help maximize the profits.

## Data Description

The insurance price forecast dataset contains historical records for 1338 insured
customers. The column definitions are below
● age: Age of the primary beneficiary.
● sex: Gender of the primary beneficiary.
● BMI: Body mass index of primary beneficiary
● children: Number of children the primary beneficiary has.
● smoker: Whether the primary beneficiary smokes.
● region: The primary beneficiary's residential area in the US.
● charges: Individual medical costs billed by health insurance.

## Tech Stack
➔ Language: Python
➔ Libraries: pandas, numpy, matplotlib, pingouin, sklearn, xgboost, skopt

## Approach
* Exploratory Data Analysis (EDA)
  - Distributions
  - Univariate Analysis
  - Bivariate Analysis
  - Correlation
    + Pearson Correlation
    + Chi-squared Tests
    + ANOVA
* Build and evaluate a baseline linear model
  - Linear regression assumptions
  - Data preprocessing
  - Model training
  - Model evaluation
    + RMSE
* Improve on the baseline linear model
  - Using Sklearn's `Pipeline` to optimize the model training process
  - Model evaluation
    +RMSE
* Comparison to the baseline model
