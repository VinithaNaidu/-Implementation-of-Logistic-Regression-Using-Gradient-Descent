## EXP NO. 05
# Implementation-of-Logistic-Regression-Using-Gradient-Descent
### Date : 05.10.23
## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6. Obtain the graph.

## Program:
```

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: D. Vinitha Naidu
RegisterNumber:  212222230175

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y== 1][:, 0], X[y==1][:,1],label="Admitted")
plt.scatter(X[y== 0][:, 0], X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  J= -(np.dot(y,np.log(h))+ np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(X.T,h-y)/X.shape[0]
  return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta, X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta, X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return J

def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min, x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min, y_max=X[:,1].min()-1,X[:,0].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot = np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)

def plotDecisionBoundary(theta,X,y):
  x_min, x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min, y_max=X[:,1].min()-1,X[:,0].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot = np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >=0.5).astype(int)

np.mean(predict(res.x,X)==y)
```
## Output:
### 1. Array Value of x:
![Screenshot 2023-10-02 125947](https://github.com/ShanmathiShanmugam/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121243595/2828d1d7-c0f3-4810-a607-7d30fcc175ee)

### 2. Array Value of y:
![image](https://github.com/ShanmathiShanmugam/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121243595/c813c999-10ca-43db-a37c-83f10d242657)
### 3. Exam 1 - score graph:
![image](https://github.com/ShanmathiShanmugam/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121243595/8e1d564a-95b8-4aae-9c05-66c96a057d06)


### 4. Sigmoid function graph:
![image](https://github.com/ShanmathiShanmugam/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121243595/0bd3452b-367a-49ef-b788-7d9b42b73b55)

### 5. X_train_grad value:
![image](https://github.com/ShanmathiShanmugam/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121243595/8639abdb-20e6-426a-a9bd-1ad23e73dd6e)

### 6 Y_train_grad value:
![image](https://github.com/ShanmathiShanmugam/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121243595/d286bec0-95a8-4a0b-bb2b-9e197806fbb3)

### 7. Print res.x:
![image](https://github.com/ShanmathiShanmugam/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121243595/1a427533-016f-4a8d-920c-ae353785c224)

### 8. Decision boundary - graph for exam score:
![image](https://github.com/ShanmathiShanmugam/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121243595/46be6842-a89f-4832-bedb-fd6c91c98a48)

### 9. Proability value:
![image](https://github.com/ShanmathiShanmugam/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121243595/4f8cdcf0-fe1d-4b29-a172-55b161622a2e)

### 10. Prediction value of mean:
![image](https://github.com/ShanmathiShanmugam/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121243595/4ded4271-8314-41a7-97b2-bd2b754edf65)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
