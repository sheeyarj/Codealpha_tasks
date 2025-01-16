#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[5]:


df = pd.read_csv('C:\\Users\\Shreeya Rajurikar\\Downloads\\Unemployment in India.csv')
df.sample(8)


# In[6]:


df.columns


# In[7]:


df[' Frequency'].value_counts()


# In[9]:


print(df['Region'].value_counts())


# In[10]:


df.isnull().sum()


# In[11]:


df.duplicated().sum()


# In[12]:


print('row count--->',df.shape[0])
print('column count--->',df.shape[1])


# In[13]:


df.dtypes


# In[14]:


df.dtypes.to_frame().rename(columns={0:"Data-Types"})


# In[15]:


df.isnull().sum().to_frame().rename(columns={0:"Total No. of Missing values"})


# In[18]:


df.rename(columns={"Region":"States"," Date":"Date"," Estimated Unemployment Rate (%)":"Estimated Unemployment Rate (%)"," Estimated Employed":"Estimated Employed"," Estimated Labour Participation Rate (%)":
                   "Estimated Labour Participation Rate (%)"},inplace=True)


df.head()


# In[19]:


df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)


# In[22]:


df[:6]


# In[23]:


df["Month"] = df["Date"].dt.month

df.head()


# In[24]:


df.head()


# In[25]:


df.drop(columns="Month",inplace=True)


# In[26]:


df["Date"].unique()


# In[27]:


df["Year"] = df["Date"].dt.year
     
df.head()


# In[28]:


df["Year"].unique()


# In[29]:


df["States"].unique()


# In[30]:


df["Area"].unique()


# In[31]:


round(df.select_dtypes(include=["float","int"]).describe().T,2)


# In[35]:


df[df["Estimated Unemployment Rate (%)"]>75]


# In[36]:


round(df.groupby(["States"])[["Estimated Unemployment Rate (%)","Estimated Employed",
                              "Estimated Labour Participation Rate (%)"]].mean(),2).sort_values(by="Estimated Unemployment Rate (%)",ascending=False)


# In[38]:


get_ipython().system('pip install seaborn --upgrade')
import seaborn as sns
import matplotlib.pyplot as plt
x = df.groupby(["States"])["Estimated Unemployment Rate (%)"].mean().sort_values(ascending=False).to_frame()
plt.figure(figsize=(10,5))

# Pass x and y as columns within the data argument
sns.barplot(data=x, x=x.index, y="Estimated Unemployment Rate (%)", palette=sns.color_palette("Blues",28)[::-1])

plt.title("State wise Average Unemployment Rate",fontweight="black",fontsize=20,pad=10)
plt.xticks(rotation=90)
plt.show()
     


# In[39]:


get_ipython().system('pip install seaborn --upgrade')
import seaborn as sns
import matplotlib.pyplot as plt
x = df.groupby(["States"])["Estimated Labour Participation Rate (%)"].mean().sort_values(ascending=False).to_frame()
plt.figure(figsize=(10,5))

# Pass x and y as columns within the data argument
sns.barplot(data=x, x=x.index, y="Estimated Labour Participation Rate (%)", palette=sns.color_palette("Blues",28)[::-1])

plt.title("State wise Average Labour Participation Rate",fontweight="black",fontsize=20,pad=10)
plt.xticks(rotation=90)
plt.show()


# In[40]:


get_ipython().system('pip install seaborn --upgrade')
import seaborn as sns
import matplotlib.pyplot as plt

x = df.groupby(["Area"])["Estimated Unemployment Rate (%)"].mean().sort_values(ascending=False).to_frame()
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)

# Pass x and y as columns within the data argument
sns.barplot(data=x, x=x.index, y="Estimated Unemployment Rate (%)", palette=sns.color_palette("Blues",3)[::-1])

plt.title("State wise Average Unemployment Rate",fontweight="black",fontsize=15,pad=10)
plt.xticks(rotation=90)

z = df.groupby(["Area"])["Estimated Labour Participation Rate (%)"].mean().sort_values(ascending=False).to_frame()
plt.subplot(1,2,2)

# Pass x and y as columns within the data argument
sns.barplot(data=z, x=z.index, y="Estimated Labour Participation Rate (%)", palette=sns.color_palette("Blues",3)[::-1])

plt.title("State wise Average Labour Participation Rate",fontweight="black",fontsize=15,pad=10)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[41]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
# Pass x and y as keyword arguments
ax = sns.lineplot(x=df["Year"], y=df["Estimated Unemployment Rate (%)"], color="Blue")
ax.set_xticklabels(["","2019","","","","","2020"])
plt.title("Unemployment Rate Over the Years",fontweight="black",fontsize=20,pad=10)

plt.subplot(1,2,2)
# Pass x and y as keyword arguments
ax = sns.lineplot(x=df["Year"], y=df["Estimated Labour Participation Rate (%)"],color="Red")
ax.set_xticklabels(["","2019","","","","","2020"])
plt.title("Labour Rate Over the Years",fontweight="black",fontsize=20,pad=10)
plt.tight_layout()
plt.show()


# In[42]:


import matplotlib.pyplot as plt
     
df.columns


# In[43]:


plt.figure(figsize=(10,5))

# Pass x and y as columns within the data argument
# Use DataFrame x instead of z
sns.barplot(data=x, x="Estimated Unemployment Rate (%)", y=x.index, palette=sns.color_palette("Blues",30)[::-1])

plt.title("Most Affected States/UT of India During the COVID-19 Lockdown in terms of Unemployment Rate",
          fontweight="black",fontsize=20,pad=20)
plt.show()


# In[44]:


plt.figure(figsize=(10,5))
# Pass x and y as keyword arguments
sns.barplot(x=z["Estimated Labour Participation Rate (%)"],y=z.index,palette=sns.color_palette("Blues",30)[::-1])
plt.title("Most Affected States/UT of India During the COVID-19 Lockdown in terms of Labour Rate",
          fontweight="black",fontsize=20,pad=30)
plt.show()


# In[ ]:




