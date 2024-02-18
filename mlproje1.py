# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 21:31:07 2024

@author: hp
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#2.veri yukleme
veriler = pd.read_csv('odev_tenis.csv')
print(veriler)

from sklearn import preprocessing
veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)
#bütün kolonlara bu laberencoder fittransform methodunu apply et
#numeric değerleri encode edilmesini istemiyoruz 
# ve ilk kolonu onehotencodinge cevirmemiz gerekecek
c = veriler2.iloc[:,:1]
from sklearn import preprocessing
ohe = preprocessing.OneHotEncoder()
c =ohe.fit_transform(c).toarray()
print(c) 

havadurumu = pd.DataFrame(data=c, index= range(14) , columns=['o','r','s'])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]], axis=1)
sonveriler = pd.concat([ veriler2.iloc[:,-2:],sonveriler],axis=1)
# windy ve playde label encoding 0 1 dönüşümü yapmış olduk
# hava durumları için onehotcoding yapmış olduk
# şimdi humidtyi tahmin edicez bunun için multilinearreggesion kullanıcaz
# humidty bağımlı değişken diğerleri bağımsız değişken

#verilerin test ve egitim icin bolunmesi
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
print(y_pred)
#bide bu sistemi iyileştirebiliriz sistemden bazı anlamsız değişkenleri sistemden çıkarabiliriz
import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values= sonveriler.iloc[:,:-1],axis=1)
x_1 = sonveriler.iloc[:,[0,1,2,3,4,5]].values
x_1 = np.array(x_1,dtype=float)
model = sm.OLS(sonveriler.iloc[:,-1:],x_1).fit() 
print(model.summary())

sonveriler =sonveriler.iloc[:,1:]

#backward eliminiation
import statsmodels.api as sm

x = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1],axis=1 )
x_1 = sonveriler.iloc[:,[0,1,2,3,4]].values
x_1 = np.array(x_1,dtype=float)
model = sm.OLS(sonveriler.iloc[:,-1:],x_1).fit() #windy colonunu attık 
print(model.summary())

x_train = x_train.iloc[:,1:] # 0. sütunu attık
x_test = x_test.iloc[:,1:]

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)













