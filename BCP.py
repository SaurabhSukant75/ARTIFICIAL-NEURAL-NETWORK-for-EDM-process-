# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 13:20:26 2018
@author: saurabhsukant75
"""
import numpy as np
import pandas as pd
class Neural_network(object):
    def __init__(self):
        #size of i/o
        self.input_size=4
        self.output_size=2
        self.hiddenlayer_size=30
        self.no_of_hiddenlayer=1
        #learning rate
        self.gamma=.5
        #weight matrix 
        self.W1=np.random.randn(self.input_size,self.hiddenlayer_size)
        self.W2=np.random.randn(self.hiddenlayer_size,self.output_size)
    def forward(self,X):
        self.z1=np.dot(X,self.W1)
        self.z2=self.sigmoid(self.z1)
        self.z3=np.dot(self.z2,self.W2)
        o=self.sigmoid(self.z3)
        return o
    def sigmoid(self,x):
        #activation function
        return 1/(1+np.exp(-x))
    def sigmoidPrime(self,x):
        #derivative of activation function.
        return x*(1-x)
        
    def backward(self, X, y, y_pred,gamma):
         # backward propgate through the network
         self.o_error = y - y_pred # error in output
         self.o_delta3 = self.o_error*self.sigmoidPrime(y_pred) # applying derivative of sigmoid to error

         self.z2_error = self.o_delta3.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
         self.z2_delta2 = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

         self.W1 +=gamma* X.T.dot(self.z2_delta2) # adjusting first set (input --> hidden) weights
         self.W2 +=gamma*self.z2.T.dot(self.o_delta3) # adjusting second set (hidden --> output) weights 
        
    def train(self,X,Y,gamma):
        y_pred=self.forward(X)
        self.backward(X,Y,y_pred,gamma)
    def predict(self,x_test):
        o=self.forward(x_test)
        return o     
    def fit(self,x_train,y_train,no_of_iteration=1000,gamma=.5):
        #for learning of algoritham
        for i in range(no_of_iteration):
            self.train(x_train,y_train,gamma)
    def accuracy(self,y_predict,y_actual):
        return  abs(100-100*np.mean(np.array(abs(y_predict-y_actual))))
    def optimum_parameter(self):
        #developed by naive approach.
        #Rcently i'm working on GENETIC ALGORITHM TO find optimum parameter.
        #AVOID THE BELOW CODE IT IS NOT A EFFICIENT MEHOD.
        possible_prams=pd.read_csv("file:///C:/Users/dell/Desktop/fem/ANN FOR EDM/vitt.csv")              
        mrr=self.predict(possible_prams/np.amax(possible_prams, axis=0))
        x=mrr[0]
        parameter=possible_prams.iloc[np.argmax(x,axis=0)]  
        #print("optimum mrr is:",np.max(x,axis=0)*100)
        print("optimum parameter is:",parameter)              
    
    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, delimiter=',',fmt="%s")
        np.savetxt("w2.txt", self.W2, delimiter=',',fmt="%s")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        