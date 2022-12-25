# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 12:39:43 2022

@author: b4100
"""
import math

def f(x):
  return -5*pow(x,4)-2*pow(x,3)-x+9

def f2(x):
  y=np.zeros(len(x))
  y[np.where(x>0.2)]=1
  return y

def f5(x):
  y=np.zeros(len(x))
  y[np.where(x>-0)]=1
  y[np.where(x>0.5)]=0
  return y

def f7(x):
    return np.sin(x)+5*np.cos(x)

'''
def F(A,x):
  
  y=0
  for i in range(int(len(A)/2)):
    y+=A[i*2]*np.sin(x*(len(A)-i-1))+A[i*2+1]*np.cos(x*(len(A)-i-1))

  return y
'''
'''
def F(A,x):
  y=0
  for i in range(len(A)):
    y+=A[i]*pow(x,len(A)-i-1)
  return y
'''

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    y=x
    if y<0: y=0
    return y

def F(A,x):
  a1=relu(A[0]*x+A[1])
  a2=relu(A[2]*a1+A[3])
  a3=relu(A[4]*a1+A[5])
  a4=sigmoid(A[6]*a2+A[7]*a3+A[8])
  a5=sigmoid(A[9]*a2+A[10]*a3+A[11])
  
  a6=sigmoid(A[12]*a4+A[13]*a5+A[14])
  
  
  return a6


#高—低

#高—低

import numpy as np
import random
X=np.linspace(-1,1)
Y=f5(X)

#X=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#Y=[5,4,3,2,1,2,3,4,5,4,3]

h=1e-4

def partialW(A,x,y,n): #偏微第n個x的係數
  A1=A.copy()
  A2=A.copy()
  A1[n]+=h
  return (pow(y-F(A1,x),2)-pow(y-F(A2,x),2))/h





def train_GD():
  for j in range(len(A)):
    for i in range(len(X)):
      x=X[i]
      y=Y[i]
      p=partialW(A,x,y,j)
      A[j]=A[j]-lr*p/batch_size

def train_SGD():
  for j in range(len(A)):
    p=0
    arr=np.zeros(len(X))
    for i in range(batch_size):
      index=random.randint(0,len(X)-1)
      while arr[index]==1:
        index=random.randint(0,len(X)-1)
      arr[index]=1
      x=X[index]
      y=Y[index]
      p+=partialW(A,x,y,j)
    A[j]=A[j]-lr*p/batch_size

def train_adam():
  global mt_1
  global vt_1
  for j in range(len(A)):
    
    
    arr=np.zeros(len(X))
    for i in range(batch_size):
      
      index=random.randint(0,len(X)-1)
      while arr[index]==1:
        index=random.randint(0,len(X)-1)
      
      #index=i
      arr[index]=1
      x=X[index]
      y=Y[index]
      g=partialW(A,x,y,j)

      mt[j]=b_1*mt_1[j]+(1-b_1)*g
      vt[j]=b_2*vt_1[j]+(1-b_2)*pow(g,2)
      _m[j]=mt[j]/(1-b_1)
      _v[j]=vt[j]/(1-b_2)
    
      vt_1[j]=vt[j]
      mt_1[j]=mt[j]
      
      A[j]=A[j]-lr*_m[j]/(math.sqrt(_v[j])+1e-8)

import matplotlib.pyplot as plt
n=15
'''
lerr=9999999

A=np.zeros(n)
batch_size=8
lr=0.1
for i in range(50000):
  train_GD()
  error=0
  for j in range(len(X)):
    x=X[j]
    y=Y[j]
    error+=pow(y-F(A,x),2)
  if error<1e-2:
    break
  lerr=error
  if i%1==0:
    Y2=np.zeros(len(X))
    for k in range(len(X)):
      x=X[k]
      Y2[k]=F(A,x)
    plt.plot(X,Y2)
    print(n-1,'階')
    print('迭代',i,'次')
    print('->',A)
    print('error=',error)
    plt.title('GD')
    plt.plot(X,Y)
    plt.ylim(min(Y)-1,max(Y)+1)
    plt.show()


A=np.zeros(n)
lerr=9999999
lr=0.1
batch_size=8
for i in range(50000):
  train_SGD()
  error=0
  for j in range(len(X)):
    x=X[j]
    y=Y[j]
    error+=pow(y-F(A,x),2)
  if abs(lerr-error)<1e-4:
    break
  lerr=error
  if i%1==0:
    Y2=np.zeros(len(X))
    for k in range(len(X)):
      x=X[k]
      Y2[k]=F(A,x)
    plt.plot(X,Y2)
    print(n-1,'階')
    print('迭代',i,'次')
    print('->',A)
    print('error=',error)
    plt.title('SGD')
    plt.plot(X,Y)
    plt.ylim(min(Y)-1,max(Y)+1)
    plt.show()
'''

lerr=9999999
b_1=0.9
b_2=0.999
lr=1e-1
batch_size=8
A=np.ones(n)
mt_1=np.zeros(len(A))
vt_1=np.zeros(len(A))
mt=np.zeros(len(A))
vt=np.zeros(len(A))
_m=np.zeros(len(A))
_v=np.zeros(len(A))


for i in range(50000):
  train_adam()
  error=0
  for j in range(len(X)):
    x=X[j]
    y=Y[j]
    error+=pow(y-F(A,x),2)
  if abs(lerr-error)<1e-5:
    break
  lerr=error
  if i%1==0:
    Y2=np.zeros(len(X))
    for k in range(len(X)):
      x=X[k]
      Y2[k]=F(A,x)
    plt.plot(X,Y2)
    print(n-1,'階')
    print('迭代',i,'次')
    print('->',A)
    print('error=',error)
    plt.title('adam')
    plt.plot(X,Y)
    plt.ylim(min(Y)-1,max(Y)+1)
    plt.show()

