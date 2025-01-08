#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="darkgrid",font_scale=1.5)

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix


# In[3]:


import pandas as pd

df = pd.read_csv('C:\\Users\\Shreeya Rajurikar\\Downloads\\Iris.csv')

print(df.head())


# In[4]:


df.tail()


# In[5]:


df_copy=df.copy()
     
df_copy    


# In[6]:


df_copy.drop(columns=['Id'],axis=0,inplace=True)
     
df_copy.dtypes


# In[9]:


df_copy.shape
print('Rows ---->',df.shape[0])
print('Columns ---->',df.shape[1])


# In[12]:


df_copy.describe()     


# In[13]:


df_copy.size  


# In[14]:


df_copy.info()


# In[15]:


df_copy.columns = ['sl','sw','pl','pw','species']
df_split_iris=df_copy.species.str.split('-',n=-1,expand=True)
df_split_iris.drop(columns=0,axis=1,inplace=True)
df_split_iris


# In[16]:


df3_full=df_copy.join(df_split_iris)
df3_full


# In[17]:


df3_full.drop(columns='species',axis=1,inplace=True) #Drop excessive column     

df3_full


# In[18]:


df3_full.shape


# In[19]:


df3_full.isna()


# In[20]:


df3_full.isna().sum()


# In[22]:


import pandas as pd
# Assuming 's' is the column with categorical data like 'setosa'
df3_full_numeric = df3_full.select_dtypes(include=['number'])  # Select only numeric columns
df3_full_numeric.corr()  # Calculate correlation matrix on numeric data


# In[23]:


df3_full.describe()


# In[24]:


df.dtypes.to_frame().rename(columns={0:"Data-Types"})


# In[25]:


df.isnull().sum().to_frame().rename(columns={0:"Total No. of Missing Values"})


# In[26]:


df.duplicated().sum()


# In[33]:


plt.figure(figsize=(10,4))
sns.countplot(x=df["Species"],data=df,palette="Set2") 
plt.title("Iris Species Distribution",pad=10)
plt.show()


# In[34]:


plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
sns.scatterplot(x="Species", y="SepalLengthCm", data=df, hue="Species", palette="Set2") # Pass 'Species' as x and 'SepalLengthCm' as y
plt.title("SepalLength  VS  Species", pad=20, fontweight="black")
plt.subplot(1,2,2)
sns.scatterplot(x="Species", y="SepalWidthCm", data=df, hue="Species", palette="Set2") # Pass 'Species' as x and 'SepalWidthCm' as y
plt.title("SepalWidth  VS  Species", pad=20, fontweight="black")
plt.show()    


# In[37]:


x = df.drop(columns=["Species","Id"])
y = df["Species"]
     

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
     

print(x_train.shape,y_train.shape)

print(x_test.shape,y_test.shape)


# In[38]:


clf = LogisticRegression()
     

clf.fit(x_train,y_train)


# In[39]:


train_pred = clf.predict(x_train)
test_pred = clf.predict(x_test)
     

print("Accuraacy on Training Data is: ",accuracy_score(y_train,train_pred)*100)

print("Accuracy on Tetsing Data is:",accuracy_score(y_test,test_pred)*100)


# In[40]:


cm = confusion_matrix(y_test,test_pred)
cm


# In[41]:


plt.figure(figsize=(4,2))
sns.heatmap(cm,annot=True,fmt="g",cmap="summer")
plt.show()


# In[ ]:




