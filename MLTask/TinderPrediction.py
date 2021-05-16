#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# In[62]:


#Reading data from csv
data=pd.read_csv('task2.csv')
data1=data


# In[63]:


#Checking for Null Values
data.isnull().sum()


# In[64]:


#Preprocessing of Data
X_cat = data.copy()
X_cat = data.select_dtypes(include=['object'])
X_enc = X_cat.copy()
X_enc = pd.get_dummies(X_enc, columns=['Segment type','Segment Description','Answer'])
data1=data1.drop(['ID','Segment type','Segment Description','Answer'],axis=1)
Finaldata=pd.concat([data1,X_enc],axis=1)


# In[65]:


#Seperating as dependant and independant variables
X=Finaldata.drop('It became a relationship',axis=1)
y=Finaldata['It became a relationship']


# In[66]:


#Spliiting data for training and testing
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=12)


# In[67]:


#Scaling down values for optimisation
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[68]:


#Starting Random Forest Classifier
rfc=RandomForestClassifier(n_estimators=300)
rfc.fit(X_train,y_train)
pred_rfc=rfc.predict(X_test)


# In[69]:


#Results Time!!!
print(classification_report(y_test, pred_rfc))


# In[70]:


#90 percent accuracy is pretty good. We can change n_estimators(hyperparameters) to get different values


# In[71]:


#Starting SVM Classifier
clf=svm.SVC()
clf.fit(X_train,y_train)
pred_clf=clf.predict(X_test)


# In[72]:


#Results Time!!!
print(classification_report(y_test,pred_clf))


# In[73]:


#Starting neural network
mlpc=MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=500)
mlpc.fit(X_train,y_train)
pred_mlpc=mlpc.predict(X_test)


# In[74]:


#Results time!!!
print(classification_report(y_test,pred_mlpc))


# In[ ]:


#We can change values of layer sizes and max iterations for different values

