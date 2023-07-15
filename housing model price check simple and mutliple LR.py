#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("Housing.csv")


# In[2]:


df.shape


# In[3]:


df


# In[4]:


##missings values


# In[5]:


df.isna().sum()


# ### treating the categorical variable
# 

# In[6]:


df.columns


# In[7]:


l1=['driveway', 'recroom', 'fullbase', 'gashw', 'airco', 'garagepl','prefarea']


# In[8]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in l1:
    df[i]=le.fit_transform(df[i])
    
df.head()    
    
    


# ###feature Scaling

# ### treating of numerical variable

# In[9]:


from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
df['lotsize']= mm.fit_transform(df[['lotsize']])
df.head()


# ### feature selection
# 

# In[10]:


from sklearn.feature_selection import SelectKBest,f_regression
x=df.iloc[:,2:]
y=df['price']
sk=SelectKBest(f_regression,k=2)
sk.fit_transform(x,y)
sk.get_support(indices=True)


# In[11]:


x=df[['lotsize','bathrms']]
y=df['price']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,shuffle=True)


# ### step-2 learning

# In[12]:


from sklearn.linear_model import LinearRegression
slr=LinearRegression()
slr.fit(xtrain,ytrain)


# In[13]:


slr.coef_       # m value


# In[14]:


slr.intercept_   # c value in y=mx+c


# ### evaluate

# In[15]:


ypred=slr.predict(xtest)
from sklearn.metrics import r2_score
r2_score(ytest,ypred)


# ### step 4 prediction

# In[16]:


xnew=[[1500,2]]
xs=mm.fit_transform(xnew) #scaled value of my predciting value


slr.predict(xs)


# ###shrinkage methods lasso regression L1 lambda lasso L2 ridge
# y-f(x)=error

# In[ ]:


from sklearn.linear_model import Lasso


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




