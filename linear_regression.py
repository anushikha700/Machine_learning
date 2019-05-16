import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_csv("C:/Users/Dell/Desktop/area_price.csv")
print(df)

reg=linear_model.LinearRegression()
reg.fit(df[['area']],df.price)

plt.xlabel("Area(sq ft)")
plt.ylabel("Price ($)")
plt.title("Area-Price")

plt.scatter(df.area,df.price,color='red',marker='+')
plt.plot(df.area,reg.predict(df[['area']]),color='blue')
plt.show()

print("Prediction : ",reg.predict([[500]]))
print("Prediction : ",reg.predict([[530],[700]]))
print("Coefficien : ",reg.coef_)
print("Intercept : ",reg.intercept_)


