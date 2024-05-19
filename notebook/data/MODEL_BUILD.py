import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Modelling

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

#PREPARING X AND Y Variable
print(os.getcwd())

df = pd.read_csv('notebook/data/StudentsPerformance.csv')
x = df.drop(columns=['math score'], axis=1)
print(x.head())
y = df['math score']
print(y.head())

#Create Column Transformers

num_features = x.select_dtypes(exclude="object").columns
cat_features = x.select_dtypes(include="object").columns

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, cat_features),
        ("StandardScaler", numeric_transformer, num_features),
    ]
)

X = preprocessor.fit_transform(x)

print(X.shape)

# Separate Train and Test

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_Train.shape, X_Test.shape)

#Evaluate Function to five metrics after model training

def evaluate_model(true,predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square

models = {
    "Linear Regression" : LinearRegression(),
    "Lasso" : Lasso(),
    "Ridge" : Ridge(),
    "K-Neighbours Regressor" : KNeighborsRegressor(),
    "Decision Tree" : DecisionTreeRegressor(),
    "Random Forest Regressor" : RandomForestRegressor(),
    "XGBRegressor" : XGBRegressor(),
    "CatBoosting Regressor" : CatBoostRegressor(verbose=False),
    "AdaBoost Regressor" : AdaBoostRegressor()
}

model_list = []
r2_list = []

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_Train, Y_Train) #Train model
    y_train_pred = model.predict(X_Train)
    y_test_pred = model.predict(X_Test)
    model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(Y_Train ,y_train_pred)
    model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(Y_Test, y_test_pred)
    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])

    print("model performance on Training set")
    print("- Root Mean Squared Error :{:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score : {:.4f}".format(model_train_r2))

    print("----------------------------------------")

    print("model performance on Test set")
    print("- Root Mean Squared Error :{:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score : {:.4f}".format(model_test_r2))

    r2_list.append(model_test_r2)
    print('='*35)
    print("\n")

    print(pd.DataFrame(list(zip(model_list,r2_list)),columns=['Model Name','R2_Score']).sort_values(by=["R2_Score"],ascending=False))

    #LINEAR REGRESSION

    lin_model = LinearRegression(fit_intercept= True)
    lin_model = lin_model.fit(X_Train,Y_Train)
    y_pred = lin_model.predict(X_Test)
    score = r2_score(Y_Test, y_pred)*100
    print("Accuracy of the model is %.2f" %score)

    #Difference between actual and predicted values

    pred_def = pd.DataFrame({'Actual Value':Y_Test,'Predicted Value':y_pred,"Difference":Y_Test-y_pred})
    print(pred_def)
