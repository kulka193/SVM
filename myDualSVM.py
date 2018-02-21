
    
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 18:48:15 2017

@author: bharg
"""

import numpy as np
import math
from cvxopt import matrix, solvers
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

def read_data(filename):
    data=np.genfromtxt(filename,delimiter=',')  
    return data



def randomize_data(X,Y):
    data=np.concatenate((X,np.expand_dims(Y,axis=1)),axis=1)
    data_perm=np.random.permutation(data)
    X_perm=data_perm[:,:-1]
    Y_perm=data_perm[:,-1]
    return X_perm,Y_perm

def fit(X,y,C):
    N,d=X.shape
    y=np.reshape(y,(N,1))
    k1=y*X
    K=np.matmul(k1,k1.T)
    P=matrix(K)
    q=matrix(-1*np.ones((N,1)))
    g1=-1*(np.eye(N))
    g2=np.eye(N)
    G=matrix(np.vstack((g1,g2)))
    h1=np.zeros((N,1))
    h2=C*np.ones((N,1))
    H=matrix(np.vstack((h1,h2))) 
    y=np.reshape(y,(1,-1))
    A=matrix(y)
    b=matrix(np.zeros((1)))
    solvers.options['show_progress']=False
    sol=solvers.qp(P,q,G,H,A,b)
    alpha=sol['x']
    alpha1=np.array(alpha)
    #print(alpha1.shape)
    w=np.zeros((d,1))
    y=np.reshape(y,(N,1))
    #count=0
    for i in range(N):
        if (alpha1[i]>=0):
            #count=count+1
            s=alpha1[i]*y[i]
            w=w+(s*np.reshape(X[i,:].T,(d,1)))
    #print(np.shape(w))
    #print(count)
    b=np.zeros((1,1))
    X1=np.zeros((1,d))
    y1=np.zeros((1,1))
    for i in range(N):
        if (alpha[i]>0 and alpha[i]<C):
            #f=np.matmul(w.T,X[i,:])
            #print(f.shape)
            #b=np.vstack((b,((1/y[i])-f)))
##
            X1=np.vstack((X1,X[i,:]))
            y1=np.vstack((y1,y[i]))
    
    X1=np.delete(X1,0,0)
    y1=np.delete(y1,0,0)
    X1_mean=np.mean(X1,axis=0)
    X1_mean=np.reshape(X1_mean,(1,len(X1_mean)))
    y1_mean=np.mean(y1)
    b=y1_mean-np.matmul(X1_mean,w)
    #b=np.delete(b,(0,0))
    #print('------')
    #print(b)
    #b=np.mean(b,axis=0)
    return w,b

def predict(X,w,b):
    N,d=X.shape
    yresult=np.zeros((N,1))
    for i in range(N):
        yresult[i]=np.sign(np.matmul(w.T,X[i,:])+b)
    return yresult


### Main function

def myDualSVM(filename,C):
    data=read_data(filename)
    m=np.size(data,0)
    n=np.size(data,1)
    X=data[:,1:n]
    y=data[:,0]
    for i in range(m):
        if y[i]==1:
            y[i]=-1
        elif y[i]==3:
            y[i]=1

    X,y=randomize_data(X,data[:,0])
    n_samples,n_features=X.shape
    k=10 ### 10 fold cross validation
    mean_error_rate=np.zeros((len(C),1))
    std_deviation_error_rate=np.zeros((len(C),1))
    
    for h in range(len(C)):
        
        error_rates=np.zeros(k)
        error_rates1=np.zeros(k)
        ### Splitting data for cross validation
        Xsplit=np.array_split(X, k)
        ysplit=np.array_split(y,k)
        for i in range(k):
            Xtest=Xsplit[i]
            ytest=ysplit[i]
            Xtrain=np.zeros((1,n_features))
            ytrain=np.zeros((1,1))
            for j in range(k):
                if j!=i:
                    Xtrain=np.vstack((Xtrain,Xsplit[j]))
                    ytrain=np.vstack((ytrain,(ysplit[j].reshape((-1,1)))))
                  
            Xtrain=np.delete(Xtrain,0,0) 
            ytrain=np.delete(ytrain,0,0) 
            Xtrain1=Xtrain
            ytrain1=ytrain
            W,b=fit(Xtrain1,ytrain1,C[h])
            result= predict(Xtrain1,W,b)
            count=0
            for m in range(len(result)):
                if (ytrain1[m]!=result[m]):
                    count=count+1
            error_rates[i]=float(count)/(len(result))

            result1= predict(Xtest,W,b)
            count=0
            for m in range(len(result1)):
                if (ytest[m]!=result1[m]):
                    count=count+1
            error_rates1[i]=float(count)/(len(result1))
            
               
        mean_error_rate[h]=np.mean(error_rates) 
        std_deviation_error_rate[h]=np.std(error_rates)
        mean_error_rate1=np.mean(error_rates1)
        std_deviation_error_rate1=np.std(error_rates1)                           
        #print "Error rates on training set with C =",
        print ("Error rates =",error_rates)
        print ("Mean of Error rates =",mean_error_rate[h])
        print ("Std deviation of Error rates =",std_deviation_error_rate[h])
        print('.......................................................')
        print ("Error rates on test set with C =", C[h])
        print ("Error rates =",error_rates1)
        print ("Mean of Error rates =",mean_error_rate1)
        print ("Std deviation of Error rates =",std_deviation_error_rate1)
        print('------------------------------------------------------------------------')

    #return mean_error_rate,std_deviation_error_rate
    log_mean_error_rate=np.log10(mean_error_rate)
    log_C=np.log10(C)
    plt.errorbar(log_C,log_mean_error_rate,std_deviation_error_rate,marker='x')
    plt.xlabel('log(C)')
    plt.ylabel('log(mean_error_rate)')
    plt.title('Plot of C vs Mean of error rates')
    plt.show()
    #plt.save()
    plt.savefig('3a_cv'+'plot1.png')

    


#### Function call
myDualSVM('MNIST-13.csv',[1e-8,1e-7,1e-6,1e-5,1e-4])




    







   
    
    


