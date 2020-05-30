#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries  
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd


# In[2]:


#importing datasets  
data_set= pd.read_csv("C:/Users/DELL/Downloads/50_Startups.csv")
data_set.head()


# In[8]:


#Extracting Independent and dependent Variable  
X = data_set.iloc[:, :-1]
y = data_set.iloc[:, 4]


# In[9]:


#Convert the column into categorical columns
states=pd.get_dummies(X['State'],drop_first=True)


# In[10]:


# Drop the state coulmn
X=X.drop('State',axis=1)


# In[11]:


# concat the dummy variables
X=pd.concat([X,states],axis=1)


# In[15]:


# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state=0)


# In[20]:


#Fitting the MLR model to the training set:  
from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(X_train, y_train) 


# In[21]:


#Predicting the Test set result;  
y_pred= regressor.predict(X_test)
y_pred


# In[23]:


from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
score


# In[ ]:




