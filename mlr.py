


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

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

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
print(mean_squared_error(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
score=r2_score(y_test,y_pred)
print(score)
# plt.scatter(Y_test,Y_pred)
# plt.show()




