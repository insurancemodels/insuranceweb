#!/usr/bin/env python
# coding: utf-8

# <p style = "font-size : 50px; color : #532e1c ; font-family : 'Comic Sans MS'; text-align : center; background-color : #bedcfa; border-radius: 5px 5px;"><strong>Insurance Fraud Detection</strong></p>

# In[50]:


# necessary imports

import pandas as pd
import numpy as np

import joblib
import sklearn
import streamlit as st
def convert_df(df):
    return df.to_csv().encode('utf-8')



st.title("Insurance Fraud Detection")

test_file = st.file_uploader("Choose a test file", type=["csv","xlsx"])
test_model=st.file_uploader("Choose model", type=["sav","plk"])


# If button is pressed
if st.button("Predict"):
    df = pd.read_csv(test_file)

    # Title


    # In[51]:





    # In[52]:


    # we can see some missing values denoted by '?' so lets replace missing values with np.nan

    df.replace('?', np.nan, inplace = True)


    # In[53]:


    df['collision_type'] = df['collision_type'].fillna(df['collision_type'].mode()[0])
    print(df['collision_type'].mode()[0])


    # In[54]:


    df['property_damage'] = df['property_damage'].fillna(df['property_damage'].mode()[0])
    print(df['property_damage'].mode()[0])


    # In[55]:


    df['police_report_available'] = df['police_report_available'].fillna(df['police_report_available'].mode()[0])
    print(df['police_report_available'].mode()[0])


    # In[56]:


    # dropping columns which are not necessary for prediction





    # In[57]:




    # In[58]:


    # separating the feature and target columns

    X = df


    # In[59]:


    # combining the Numerical and Categorical dataframes to get the final dataset


    # splitting data into training set and test set
    # extracting categorical columns
    cat_df = X.select_dtypes(include = ['object'])
    cat_df = pd.get_dummies(cat_df, drop_first = True)
    num_df = X.select_dtypes(include = ['int64'])

    X = pd.concat([num_df, cat_df], axis = 1)


    # In[60]:





    # In[63]:



    # some time later...

    # load the model from disk
    loaded_model = joblib.load(test_model)
    result=loaded_model.predict(X)
    result = pd.DataFrame(result, columns = ['results'])
    result["results"].replace(['Y', 'N'],
                        ["Fraud Detected", "No Fraud"], inplace=True)

    st.write(result)
    df["results"]=result
    download_csv = convert_df(df)
    st.download_button(
         label="Download File with predictions",
         data=download_csv,
         file_name='insurance_fraud_prediction.csv',
         mime='text/csv',
     )






# In[ ]:
