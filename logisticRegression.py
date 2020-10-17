# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 08:26:19 2020

@author: zpata
"""
import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
def loss_Function(y,yP):
    if(yP==0):
        yP+=0.00001
    return (y*math.log(yP))-((1-y)*math.log(1-yP))
count=0
while(True):
    X=np.random.rand(3,100)
    Y=np.random.rand(100,1)
    b=np.random.rand();
    lR=0.5
    
    weights=np.random.rand(3,1)
    z=weights.T.dot(X)+b;
    
    sigmoid_v=np.vectorize(sigmoid)
    loss_v=np.vectorize(loss_Function)
    
    A=sigmoid_v(z);
    dz=A.T-Y
    
    db=np.sum(dz)/300
    dw=X.dot(dz)/300
    weights-=lR*dw
    b-=lR*db
    
    loss=np.sum( loss_v(Y.T,A))/300
    count+=1
    if(loss<0.001):
        break;
    
    
print(loss,count);    
# aMean=np.sum(loss)/100




# print(loss)






