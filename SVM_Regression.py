import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.svm import SVC

data=pd.read_csv("Titanic-Dataset.csv")
print(data.isnull().sum())
df=data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked','SibSp','Parch'],axis=1)
df['Age'].fillna(df['Age'].mean(),inplace=True)

le=LabelEncoder()
df['Sex']=le.fit_transform(df['Sex'])

X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

st=StandardScaler()
X_sc=st.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

m=SVC(kernel="rbf")
m.fit(X_train,y_train)
y_pred=m.predict(X_test)

plt.scatter(X_sc[:,0],X_sc[:,1],c=y, cmap='coolwarm', edgecolors='k')
plt.xlabel('Pclass')
plt.ylabel('Sex')
plt.title('SVM Classification')
plt.show()