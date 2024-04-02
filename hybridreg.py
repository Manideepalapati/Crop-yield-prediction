
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import *
from sklearn.tree import DecisionTreeRegressor
import xgboost as xb
from sklearn.linear_model import LinearRegression
crop=pd.read_excel("crop yeild data.xlsx")
#print(tesla.columns)
x=crop[['Rain Fall (mm)','Fertilizer(urea) (kg/acre)','Temperature (°C)','Nitrogen (N)','Phosphorus (P)','Potassium (K)']][0:99]
y=crop[['Yeild (Q/acre)']][0:99]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=17)

from sklearn.preprocessing import MinMaxScaler

scalar=MinMaxScaler()
x_train=scalar.fit_transform(x_train)
x_test=scalar.transform(x_test)

from sklearn.ensemble import *
# regressor=StackingRegressor()
estimators = [
('dt', DecisionTreeRegressor(criterion='poisson',max_depth=4,max_features='sqrt',splitter='best',random_state=0)),
('gb',GradientBoostingRegressor(random_state=0,n_estimators=100)),
# ('MLR',LinearRegression()),

]
regressor = StackingRegressor(
    estimators=estimators,
    #  final_estimator=RandomForestRegressor(criterion='poisson',max_depth=8,max_features='log2',random_state=0,n_estimators=150)
 )

regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
print(mean_squared_error(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
score=r2_score(y_test,y_pred)
print(score)