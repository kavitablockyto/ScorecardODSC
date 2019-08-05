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

# Data Description
Collection_sample.describe()

   
