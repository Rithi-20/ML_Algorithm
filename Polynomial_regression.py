import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data=pd.read_csv("Titanic-Dataset.csv")
df=pd.DataFrame(data)
df=df[['Age','Fare','Pclass']].dropna()

X=df[['Fare','Pclass']]
y=df['Age']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

st=StandardScaler()
X_train_sc=st.fit_transform(X_train)
X_test_sc=st.transform(X_test)

poly=PolynomialFeatures(degree=2)
X_train_poly=poly.fit_transform(X_train_sc)    
X_test_poly=poly.transform(X_test_sc)

model=LinearRegression()
model.fit(X_train_poly,y_train)
y_pred=model.predict(X_test_poly)

plt.scatter(X_test['Fare'],y_test,color='blue',label='Actual Data')
sorted_indices = np.argsort(X_test['Fare'])
sorted_fare = X_test['Fare'].iloc[sorted_indices]
sorted_pred = y_pred[sorted_indices]
# plt.scatter(X_test['Fare'],y_pred,color='red',label='Predicted Data')

plt.plot(sorted_fare, sorted_pred, color='red', linewidth=2, label=f'Polynomial Regression (degree=2)')

plt.xlabel('Fare')
plt.ylabel('Age')
plt.title('Polynomial Regression')
plt.legend()
plt.show()


