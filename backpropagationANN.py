# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 18:08:49 2018

@author: saurabhsukant75
"""
from BCP import Neural_network
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
df=pd.read_csv("file:///C:/Users/dell/Desktop/ppt/fem/ANN FOR EDM/edm-data.csv")
#df["creater_depth"]=df["CD(μm)"]
X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:59,0:4], 
                        df.iloc[:59,4:6], test_size=0.33, random_state=4)
#normalization
scalerX = MaxAbsScaler()
scalery = MaxAbsScaler()
# fit and transform
X_train = scalerX.fit_transform(X_train)
Y_train = scalery.fit_transform(Y_train)
X_test = scalerX.transform(X_test)
Y_test = scalery.transform(Y_test)
NN = Neural_network()
#learning of algorithm
NN.fit(X_train, Y_train,1000,.3)
#prediction on test set
op=NN.predict(X_test)

sss=["MRR(normalised)","crater_depth(normalised)"]
#mean square error
error=np.array(abs(op-Y_test)).T
#calculating % error in each
error_in_mrr=[] # %error in mrr
for i,j in zip(error[0],Y_test[:,0]):
    error_in_mrr.append(i/j*100)
error_in_craterDepth=[] # %error in mrr
for i,j in zip(error[1],Y_test[:,1]):
    error_in_craterDepth.append(i/j*100)   
    
error_matrix_in_percent=np.array([error_in_mrr,error_in_craterDepth],dtype=int)  
test=np.array(Y_test)
op=np.transpose(op)
#for plotting the graph
for i in range(1,-1,-1):
     index=np.arange(len(Y_test))
     plt.figure(figsize=(10,3))
     plt.bar(index,error_matrix_in_percent[i],color="r",width=.25,label="error")
     plt.xticks(index,index)
     plt.ylabel('error(in %)')
     plt.title("absolute error in each data")
     plt.legend()
     plt.tight_layout()
     plt.show()
     plt.figure(figsize=(10,3))    
     plt.bar(index,test[:,i],width=0.25, color='b',label='actual')
     plt.bar(index+.25,op[0,:],width=0.25, color='g',label='predicted')
     plt.xticks(index,index)
     plt.ylabel(sss[i])
     plt.title("ACTUAL vs PREDICTED on test data")
     plt.legend()
     plt.tight_layout()
     plt.show()
    

print("ACCURACY:",NN.accuracy(NN.predict(X_train),Y_train))
def output(test_x):
        out_put=NN.predict(scalerX.transform(np.array(test_x).reshape(1,-1)))
        print("expected  normalised :[MRR,crater_depth]=",out_put)
         












