import pandas as pd
import numpy as np
import random as rnd
import os
import copy
import pickle


def load_obj(name):
    with open(name+'.pkl', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name):
    with open(name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
   


dt = {}
med_df = pd.read_csv('eICU/medication.csv')
col_names =  med_df.columns
df = med_df.groupby(['patientunitstayid'], as_index=False).apply(pd.DataFrame.sort_values, 'drugstartoffset').reset_index()
df.drop(columns=['level_0', 'level_1'], inplace=True)
df = df.dropna(subset=['drugname'])
feverDrugs = ['paracetamol',  'acetaminophen', 'ibuprofen',  'tylenol', 'advil',  'aspirin',  'motrin',  'naproxen', 'aleve', 'bayer', 'aspirin',  'ibu']
for i in range(len(df)):

    if dt.get(df.iloc[i]['patientunitstayid'])=="None":
        
        drug = (df.iloc[i]['drugname'].split()[0]).lower()
       
        if drug in feverDrugs:
            dt[df.iloc[i]['patientunitstayid']] = [df.iloc[i]['drugstartoffset']]

print('Done')

