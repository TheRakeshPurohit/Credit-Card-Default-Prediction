<h1 align="center">
  Credit Card Default Prediction System
</h1>

<h3 align="center">
    A System which is trained to lower the Frauds in Credit Cards. 
</h3>

## ✨ Overview!

- A High Performing Credit Card Default Predicted with over 91.88975 score. 
- Extensively Fine tuned state of the art machine learning models. 
- Extensively done Data Processing, Ingestion, Feature Engineering 
- Built the state of the art models using StackNet, TabNet.

## Requirements

- Python [3.7 or above] - https://www.python.org/downloads/
- Git - https://git-scm.com/download/
- Pipenv [Python package for virtual envs] - Install it using `pip install pipenv` 
- sklearn 
- pandas 
- numpy 
- pytorch-tabnet
- stacknet

## Project and code structure information

- src 
   - data_ingestion :- Ingestion of the data and some data utilities classes. 
   - validate_data:- Dataset development and outlier detection classes.
   - process_data:- data_cleaning, handling imbalanced_data, imputation of missing values, scaling numerical features and etc. 
   - feature_engineering:- Feature addition, interaction terms, mean encoding of the categorical features.
   - model_dev:- developed high performing models, trained on gpu's, hyperparametric optimization using optuna. 
   - evaluate_models:- evaluation of models on various metric such as precision, recall, AUC and ROC Curve and etc. 
   - TabNet, Autoencoder:- Deep Learning Models 

## 🙌 Show your support

Be sure to leave a ⭐️ if you like the project!

<div align="center">Made by Ayush Singh with ❤</div>
