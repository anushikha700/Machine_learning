#linear regression with one variable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_csv("C:/Users/Dell/Desktop/area_price.csv")
print(df)

reg=linear_model.LinearRegression()
reg.fit(df[['area']],df.price)

plt.xlabel("Area(sq ft)") # label x-axis in graph
plt.ylabel("Price ($)")# label y-axis in graph
plt.title("Area-Price")

plt.scatter(df.area,df.price,color='red',marker='*')
plt.plot(df.area,reg.predict(df[['area']]),color='blue')
plt.show()

print("Prediction : ",reg.predict([[500]]))  # Prediction :  [16160.10638298]
print("Prediction : ",reg.predict([[530],[700]]))] #Prediction :  [17023.40425532 21915.42553191]
print("Coefficient : ",reg.coef_) # Coefficient :  [28.77659574]
print("Intercept : ",reg.intercept_) #Intercept :  1771.8085106382969


