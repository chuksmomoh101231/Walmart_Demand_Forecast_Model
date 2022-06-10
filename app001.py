#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML


# In[3]:


h2o.init(max_mem_size=2,nthreads=-1)


# In[6]:


automl = h2o.upload_mojo('XGBoost_1_AutoML_1_20220609_171306.zip')


file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])


if file_upload is not None:
    data = pd.read_csv(file_upload)
    #data = data.fillna(0)
    data['cum_moving_average']=data['sales'].expanding().mean()
    data['exp_weighted_moving_average']=data['sales'].ewm(span=28).mean()
    data['total_price'] = data['sales'] * data['sell_price']
    #data['Moving_Average']= data['Weekly_Sales'].rolling(window=7, min_periods=1).mean()
    data = h2o.H2OFrame(data)
    predictions = automl.predict(data).as_data_frame()
    #predictions = automl.predict(data).as_data_frame()
    predictions = predictions.join(data.as_data_frame())
    st.write(predictions)
    
    @st.cache
    
    def convert_df(df):
        return df.to_csv(index = False, header=True).encode('utf-8')
    csv = convert_df(predictions)
    
    st.download_button(label="Download data as CSV",data=csv,
                file_name='credit_risk_prediction.csv',mime='text/csv')
    

        
            
            
            


# In[5]:


# FOR SINGLE PREDICTIONS, TRY ST.WRITE INSTEAD OF ST.SUCCESS


# In[ ]:




