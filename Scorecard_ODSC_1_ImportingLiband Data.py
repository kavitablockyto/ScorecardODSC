# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 09:43:55 2019

@author: KAVITA DWIVEDI
"""

#Importing the Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Setting the Working Library through File Explorer and Save option

#Reading the Dataset

df = pd.read_excel("Collection_Home3.xlsx")
Collection_sample = df.sample(frac=0.1)

# Counting Target Variable numbers
df.TARGET.value_counts()
Collection_sample.TARGET.value_counts()

#Descriptive Stats
#Target Variable Count in data and sample

df.TARGET.value_counts()/len(df)
Collection_sample.TARGET.value_counts()/len(Collection_sample)

Out[25]: 
0    0.916975
1    0.083025
Name: TARGET, dtype: float64

# Data Description
Collection_sample.describe()

    
    Out[26]: 
         DAYS_BIRTH  DAYS_EMPLOYED  DAYS_REGISTRATION  DAYS_ID_PUBLISH  \
count  27823.000000   27823.000000       27823.000000     27823.000000   
mean   16199.716673   70432.359631        5012.089279      3021.191784   
std     4329.585790  141594.192178        3552.842991      1494.991128   
min     7713.000000       6.000000           0.000000         1.000000   
25%    12594.000000     962.000000        2003.000000      1771.000000   
50%    15940.000000    2310.000000        4519.000000      3290.000000   
75%    19808.000000    6259.000000        7513.500000      4307.000000   
max    25187.000000  365243.000000       19706.000000      6383.000000   

       FLAG_EMP_PHONE  FLAG_WORK_PHONE    FLAG_PHONE  REG_CITY_NOT_LIVE_CITY  \
count    27823.000000     27823.000000  27823.000000            27823.000000   
mean         0.812565         0.205585      0.283183                0.078245   
std          0.390267         0.404136      0.450553                0.268561   
min          0.000000         0.000000      0.000000                0.000000   
25%          1.000000         0.000000      0.000000                0.000000   
50%          1.000000         0.000000      0.000000                0.000000   
75%          1.000000         0.000000      1.000000                0.000000   
max          1.000000         1.000000      1.000000                1.000000   

       REG_CITY_NOT_WORK_CITY  DAYS_LAST_PHONE_CHANGE        TARGET  
count            27823.000000            27823.000000  27823.000000  
mean                 0.229558              981.599720      0.083025  
std                  0.420556              834.132241      0.275925  
min                  0.000000                0.000000      0.000000  
25%                  0.000000              280.000000      0.000000  
50%                  0.000000              784.000000      0.000000  
75%                  0.000000             1597.000000      0.000000  
max                  1.000000             3938.000000      1.000000  
