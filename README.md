# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.
 
## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: SAHANA AS
```
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### Data Head:
![head](https://user-images.githubusercontent.com/93427923/173077929-279a193a-55f7-4de7-b705-e7260abc5290.png)

### Data Info:
![info](https://user-images.githubusercontent.com/93427923/173077947-8ca5a120-b620-4691-8485-70c09b0e6255.png)

### Data isnull():
![isnull](https://user-images.githubusercontent.com/93427923/173077964-17c3b6d5-931d-4119-8c3e-48814bdfe88b.png)

### y_pred:
![ypred](https://user-images.githubusercontent.com/93427923/173077974-78d5cb5d-6b93-4039-9dfe-34f08aee366b.png)

### Accuracy:
![accracy](https://user-images.githubusercontent.com/93427923/173077981-f93e4363-a74f-4488-80ec-c0142756fbe4.png)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

