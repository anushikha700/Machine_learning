import pandas
import pandas as pd
import numpy as np
from sklearn import linear_model

df=pd.read_csv("C:/Users/Dell/Desktop/multi_linear_reg.csv")

import math
# filling missing values  with median of features.
median_exper=math.floor(df.experience.median()) 
df.experience=df.experience.fillna(median_exper)
med_test=math.floor(df['test_score(out of 10)'].median())
df['test_score(out of 10)']=df['test_score(out of 10)'].fillna(med_test)
reg=linear_model.LinearRegression()
reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])
print("Coefficient= ",reg.coef_)
print("Intercept= ",reg.intercept_)
print("Predicted value = ",reg.predict([[11.0,7.0,9]]))
