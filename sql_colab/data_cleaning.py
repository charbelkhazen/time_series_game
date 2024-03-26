#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import pytz
# ### Data Preprocessing

# ### DATA REQUESTS REQUIREMENTS - OPTIMIZING NUMBER OF REQUESTS   - This code is comprised of a static code for historical prices + a dynamic data request code that updates everyday.
# Problem : The data can be imported as batches of 7 days for 1min interval - for a period of 30 days , I thus import 4 batches of 7 and a batch of 2.
# Then 2 minute interval can be imported in batches of 60 days. I import 1 batch of 30 days .
# Then I import (730-60) data with interval 1h
# Then the rest is imported by using 1d interval

# In[2]:


ticker ="AAPL"
working_days = yf.Ticker(ticker).history(period='max', interval='1d').index.tolist()
first_day, last_day = working_days[0], working_days[-1]
num_working_days = ( last_day  -  first_day  ).days + 1


# In[3]:


new_york_timezone = pytz.timezone('America/New_York')
today = datetime.now(new_york_timezone)


# In[4]:


last_index = int((today-first_day).days)
ordered_list = [0,6,7,13,14,20,21,27,28,29,30,59,60,729,730]   #this is the max indeces permitted


# In[5]:


def insert_into_ordered_list(ordered_list, x):
    for i in range(len(ordered_list)):
        if x <= ordered_list[i]:
            ordered_list.insert(i, x)
            truncated_list = ordered_list[:i + 1]
            return truncated_list
            break
    else:
        ordered_list.append(x)
        return ordered_list

indeces_list = insert_into_ordered_list(ordered_list, last_index)


# In[6]:


date_list = [today - timedelta(days=x) for x in range((today - first_day).days + 1)]    #!! from today to past values


# In[7]:


pairs_list = [date_list[i].strftime("%Y-%m-%d") for i in indeces_list[::-1]]


# In[8]:


def associate_interval(index):
    if index <= 29:
        return "1m"
    elif 29 < index <= 60:
        return "2m"
    elif 60 < index <= 729:
        return "1h"
    else:
        return "1d"

associated_interval_list = [associate_interval(a) for a in indeces_list[::-1]]
#This is a list of same size of pairs_list . it matches the interval of each date


# In[9]:


#Accomodate for the worst case senario - this is rare
if (len(pairs_list)%2 != 0) :
    pairs_list.append(pairs_list[-1])


# In[10]:


pairs_interval_array = np.array([pairs_list[i:i+2] + [associated_interval_list[i]] for i in range(0, len(pairs_list), 2)])


# In[11]:


stock =  yf.Ticker(ticker)


# In[12]:


base_df = pd.DataFrame()
for row in pairs_interval_array:
    start_date, end_date, interval = row
    stock = yf.Ticker(ticker)
    added_df = stock.history(start=start_date, end=end_date, interval=interval)
    base_df = pd.concat([base_df, added_df]).sort_index(ascending=False)


