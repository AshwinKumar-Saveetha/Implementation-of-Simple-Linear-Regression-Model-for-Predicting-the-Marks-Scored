# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Ashwin Kumar A
RegisterNumber:212223040021

```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE : ",rmse)
```

## Output:
![image](https://github.com/user-attachments/assets/22dc8a7d-e5b3-49a3-92c6-23a04757dd6e)
![image](https://github.com/user-attachments/assets/ac13ebcf-080b-4563-975b-b6a8cb7af031)
![image](https://github.com/user-attachments/assets/0fed736c-ac92-49c4-9cbc-64d82f0226ac)
![image](https://github.com/user-attachments/assets/54cdb72c-a7a5-4598-bb02-11fac79042db)
![image](https://github.com/user-attachments/assets/7a669a43-64e0-47ee-9904-9a4c7a45c1ff)
![image](https://github.com/user-attachments/assets/9fa19eb9-725c-4703-9e73-32efeef988c3)
![image](https://github.com/user-attachments/assets/61716ac5-6d26-4370-8420-07bac6af7f1f)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
