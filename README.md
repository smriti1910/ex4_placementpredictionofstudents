# ex4_placementpredictionofstudents


## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required packages and print the present data
2. Print the placement data and salary data
4. Using logistic regression find the predicted values of accuracy, confusion matrices
3. Find the null and duplicate values
5. Display the results
   
## Program:
```
/*
Program for Implementation of Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SMRITI .B
RegisterNumber:  212221040156
*/
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```


## Output:
![Output1](https://github.com/smriti1910/ex4_placementpredictionofstudents/assets/133334803/78ae629f-97bd-44a4-bd30-1507e704f774)
![Output2](https://github.com/smriti1910/ex4_placementpredictionofstudents/assets/133334803/32583b6c-e007-4cef-9433-3e7c35d78ff4)
![Output3](https://github.com/smriti1910/ex4_placementpredictionofstudents/assets/133334803/85a3b273-0a2f-493e-b869-d1b11e865667)
![Output4](https://github.com/smriti1910/ex4_placementpredictionofstudents/assets/133334803/782fbf44-adc6-47c4-8db7-049f9531f275)
![Output5](https://github.com/smriti1910/ex4_placementpredictionofstudents/assets/133334803/d8d06a82-f5d2-4877-a699-f16bfe69d2f8)
![Output6](https://github.com/smriti1910/ex4_placementpredictionofstudents/assets/133334803/5e068bcf-31ad-4e66-b283-283203743a84)
![Output7](https://github.com/smriti1910/ex4_placementpredictionofstudents/assets/133334803/5d5c4941-da7a-49da-91de-05861e161966)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
