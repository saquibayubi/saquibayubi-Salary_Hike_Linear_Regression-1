# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:53:03 2022

@author: Saquib_Ayubi
    
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sal=pd.read_csv("C:\\Users\\Hi\\Desktop\\Python Datasets\\Salary_Data.csv")


X=sal.iloc[:,:-1]
Y=sal.iloc[:,-1:]

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=9)

from sklearn.linear_model import LinearRegression
model2=LinearRegression()
model2.fit(x_train,y_train)

y_pred=model2.predict(x_test)


import pickle
pickle.dump(model2,open('model.pkl','wb'))



#loading model to compare the results 

model=pickle.load(open('model.pkl','rb'))
print(model.predict([[2]]))

