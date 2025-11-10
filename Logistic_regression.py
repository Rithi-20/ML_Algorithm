import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

data=load_iris()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target

print(df.head())
X=df.drop('target',axis=1)
y=df['target']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

st=StandardScaler()
X_train_sc=st.fit_transform(X_test)
X_test_sc=st.transform(X_test)
l=LogisticRegression()

l.fit(X_train,y_train)
y_pred=l.predict(X_test)

plt.scatter(X_test_sc[:,0],X_test_sc[:,1],c=y_pred,cmap='coolwarm')

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')    
plt.title('Logistic Regression')
plt.legend()
plt.show()