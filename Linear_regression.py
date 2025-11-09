import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import matplotlib.pyplot as plt

data = {
    'Experience_Level': ['Junior', 'Mid', 'Senior', 'Senior', 'Junior', 'Mid'],
    'Years': [1, 3, 5, 7, 2, 4],
    'Salary': [30000, 50000, 70000, 90000, 35000, 60000]
}

df=pd.DataFrame(data)

le=LabelEncoder()
df['Experience_Level']=le.fit_transform(df['Experience_Level'])

st=StandardScaler()
df[['Years','Experience_Level']]=st.fit_transform(df[['Years','Experience_Level']])

X=df[['Years','Experience_Level']]
y=df['Salary']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

# plt.scatter(X_test['Years'], y_test, color='blue', label='Actual Data')
# plt.plot(X_test['Years'], y_pred, color='red', label='Predicted Line')
plt.scatter(df['Years'],df['Salary'],color='blue',label='Actual Data')
plt.plot(X_test['Years'],y_pred,color='red',label='Predicted Data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression')
plt.legend()
plt.show()