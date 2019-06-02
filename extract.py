#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests
import io
import pandas as pd


# In[3]:


base_url = 'https://www.alphavantage.co/query' # data extraction is specifically designed to extract from AlphaVantage
function = '?function=' #The time series of your choice FX_DAILY, FX_WEEKLY, FX_MONTHLY
from_symbol = '&from_symbol=' # conversion from
to_symbol = '&to_symbol=' # conversion to
outputsize = '&outputsize='
apikey = '&apikey=' # personal key
datatype = '&datatype=' # csv, json etc.,

def fetch_data(functional_value,conversion_from, conversion_to, out_size, key, format_type):
    
    url = base_url
    url = url + function + functional_value
    url = url + from_symbol + conversion_from
    url = url + to_symbol + conversion_to
    url = url + outputsize + out_size
    url = url + apikey + key
    url = url + datatype + format_type
    
    r = requests.get(url, allow_redirects=True)
    df = None
    if r.ok:
       data = r.content.decode('utf8')
       df = pd.read_csv(io.StringIO(data))
    else:
        print('Issue with the url - ', url)
    
    if isinstance(df, pd.DataFrame):
       df.to_csv(conversion_from+'_'+conversion_to+'.csv',index = False, header=True)
    else:
       print('Currently only csv format is supported')


# In[6]:


# Conversion fro EUR to USD
fetch_data(functional_value='FX_DAILY', 
           conversion_from='USD',
           conversion_to='CHF',
           out_size = 'full',
           key='98B2HWGLJY5A76IV',
           format_type='csv')


# In[21]:


# Conversion fro USD to IND
fetch_data(functional_value='FX_DAILY', 
           conversion_from='USD',
           conversion_to='JPY',
           out_size = 'full',
           key='98B2HWGLJY5A76IV',
           format_type='csv')


# In[19]:


# Conversion fro USD to IND
fetch_data(functional_value='FX_DAILY', 
           conversion_from='USD',
           conversion_to='CAD',
           out_size = 'full',
           key='98B2HWGLJY5A76IV',
           format_type='csv')


# In[ ]:




