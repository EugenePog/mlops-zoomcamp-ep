#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[13]:


import pickle
import pandas as pd
from sklearn import metrics
from numpy import std


# In[4]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[5]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[6]:


df = read_data('../../data/taxi/yellow_tripdata_2022-02.parquet')


# In[7]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[14]:


#Question 1: Standard deviation for February 2022 Yellow dataset
std(y_pred)


# In[20]:


year = 2022
month = 2
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[23]:


df['prediction'] = y_pred.tolist()


# In[26]:


df_result = df[['ride_id', 'prediction']]


# In[27]:


df_result


# In[28]:


with open('predictions', 'wb') as output_file:
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


# In[ ]:




