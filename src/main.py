import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from category_encoders import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV


data = pd.read_csv('insurance.csv')

X = data.drop('charges', axis = 1)
y = data['charges']


# Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)

# Removing unnecessary columns

drop_col = ['children','sex','region']

X_train.drop(drop_col, axis = 1, inplace = True)
X_test.drop(drop_col, axis = 1, inplace = True)

# Encoding

ohe = OneHotEncoder(use_cat_names = True)
X_train = ohe.fit_transform(X_train)
X_test = ohe.transform(X_test)

X_train.drop('smoker_no', axis=1, inplace = True)
X_test.drop('smoker_no', axis=1, inplace=True)


#Recrusive Feature Elimination
rfe = RFE(estimator = XGBRegressor())
xgb = XGBRegressor()

steps = [
    ('rfe',rfe),
    ('xgb',xgb)
]


pipe = Pipeline(steps)

#Defining parameters related to each step of the pipeline
num_features = X_train.shape[1]

search_spaces = {
    'rfe__n_features_to_select' : Integer(1,num_features),  #number of features to be selected
    'xgb__n_estimators' : Integer(1,500),  #number of trees built by XGBoost
    'xgb__max_depth' : Integer(2,8),  #Max depth of trees built by XGBoost
    'xgb__reg_lambda' : Integer(1,200), #Regularization parameter (lambda) to prevent overfitting
    'xgb__learning_rate' : Real(0,1), #Learning rate of the model
    'xgb__gamma' : Real(0,2000) #Parameter for pruning the decision trees
}


xgb_bs_cv = BayesSearchCV(
    estimator = pipe, #Pipeline
    search_spaces = search_spaces,  #Search Spaces
    scoring = 'neg_root_mean_squared_error',  #Scoring metric used by BayesSearchCV for optimization
    n_iter = 70,  #number of iterations for optimization
    cv = 3, #number of folds for cross-validation
    verbose = 1,  #Progress display
    random_state = 0  #Esnuring reproducible results
)


xgb_bs_cv.fit(X_train, y_train)