#!/usr/bin/env python
# coding: utf-8

# In[14]:


#Importing the Libraries
import numpy as np
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import SVC  


# In[6]:


#Read the Dataset
df=pd.read_csv("E:\\NMDS\\flightdata1.csv")
df.head()


# In[36]:


#Tofind the shape of the dataset
df.shape


# In[35]:


#Tofind the datatype of the dataset
df.info()


# In[38]:


#to find the null values in the dataset
df.isnull()


# In[39]:


#to find the total numbers of null values in the dataset
df.isnull().sum()


# In[44]:


sns.distplot(df.MONTH)


# In[51]:


sns.countplot(df.CANCELLED)


# In[52]:


sns.heatmap(df.corr())


# In[8]:


#Univariate Analysis
df.hist()
plt.show()


# In[9]:


plt.figure(figsize = (10, 6), dpi = 100)
# setting the different color palette
color_palette = sns.color_palette("Accent_r")
sns.set_palette(color_palette)

sns.countplot(x = "DEP_DELAY", data = df)

plt.show()


# In[11]:


#Data Visualization  Distribution of CGPA
plt.figure(figsize = (10, 6), dpi = 100)
grp = dict(df.groupby('ACTUAL_ELAPSED_TIME').groups)

m = {}

for key, val in grp.items():
    
    if key in m:
        m[key] += len(val)
        
    else:
        m[key] = len(val)

    
plt.title("Distribution of ACTUAL_ELAPSED_TIME")
plt.pie(m.values(), labels = m.keys())
plt.show()


# In[12]:


#Exploratory Data Abnliysis
#Data Visualization count of ARR_DELAY
plt.figure(figsize = (10, 6), dpi = 100)
color_palette = sns.color_palette("Accent_r")
sns.set_palette(color_palette)
sns.countplot(x = "ARR_DELAY", data = df)


# In[13]:


#Data Visualization count of CRS_DEP_TIME
plt.figure(figsize = (10, 6), dpi = 100)
color_palette = sns.color_palette("cool")
sns.set_palette(color_palette)
sns.countplot(x = "CRS_DEP_TIME", data = df)
plt.show()


# In[15]:


df.skew()


# In[18]:


df=pd.read_csv("E:\\NMDS\\flightdata1.csv")
df.describe()


# In[20]:


df.isna().sum()


# In[23]:


df.duplicated().sum()


# In[ ]:




