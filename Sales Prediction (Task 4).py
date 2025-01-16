#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[2]:


df = pd.read_csv('C:\\Users\\Shreeya Rajurikar\\Downloads\Advertising.csv')


# In[3]:


df.head(10)


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


df.columns


# In[8]:


df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)


# In[9]:


df


# In[10]:


df.info()


# In[14]:


sns.set()   

df['TV'].hist()


# In[15]:


df['Newspaper'].hist()


# In[16]:


df['Radio'].hist()


# In[17]:


sns.pairplot(df,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',kind='scatter')


# In[18]:


sns.heatmap(df.corr(),annot=True,cmap='coolwarm')


# In[19]:


sns.lmplot(x='TV', y='Sales', data=df)
sns.lmplot(x='Radio', y='Sales', data=df)
sns.lmplot(x='Newspaper',y= 'Sales', data=df)


# In[27]:


corrmat = df.corr()
f, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(corrmat, vmin=0, vmax=1, square=True, cmap="YlGnBu", ax=ax)
plt.show()


# In[22]:


X=df.drop(columns='Sales')
     
Y=df['Sales']  

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=3) 

model=LinearRegression()
     
model.fit(X_train,Y_train)


# In[23]:


prediction=model.predict(X_test)     

prediction


# In[24]:


model.intercept_


# In[25]:


model.coef_


# In[26]:


accuracy_score=model.score(X_test,Y_test)*100  

print(f"Accuracy of model: {accuracy_score}%")


# In[ ]:




