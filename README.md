# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Import Libraries**: Load necessary libraries.
2. **Load Dataset**: Read the CSV file.
3. **Prepare Data**: Separate features (X) and target (Y).
4. **Split Data**: Divide into training and testing sets.
5. **Train Model**: Fit a linear regression model on training data.
6. **Make Predictions**: Predict using the test set.
7. **Visualize**: Plot results with regression lines.
8. **Evaluate**: Calculate and print MSE, MAE, RMSE.

### End of Algorithm

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

![image](https://github.com/user-attachments/assets/025a3a95-70e7-4064-ab61-f99684f190e1)
![image](https://github.com/user-attachments/assets/c1ee979a-2e3f-4a68-9042-e6ceca37d7ae)
![image](https://github.com/user-attachments/assets/9e0a8986-0911-49d9-bde3-3c3345662a1b)
![image](https://github.com/user-attachments/assets/518f5b4b-4267-4816-9e87-76c5c263d1d2)
![image](https://github.com/user-attachments/assets/078d45fc-6837-4c9d-a339-f8d0801b644c)
![image](https://github.com/user-attachments/assets/7a669a43-64e0-47ee-9904-9a4c7a45c1ff)
![image](https://github.com/user-attachments/assets/9fa19eb9-725c-4703-9e73-32efeef988c3)
![image](https://github.com/user-attachments/assets/c33a46d2-fb82-4544-a3e8-7f0e1ea9be72)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
