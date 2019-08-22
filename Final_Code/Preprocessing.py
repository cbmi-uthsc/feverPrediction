import pandas as pd
import numpy as np
import random as rnd
import os
import copy
import pickle

# visualization


# Preprocessing
from keras import backend as K
from sklearn import preprocessing
import datetime

def load_obj(name):
    with open(name+'.pkl', 'rb') as f:
        return pickle.load(f)

# Feature extraction

def feature_fun(col, df):
    
    standard_devaition = df[col].std()
    kurtosis = df[col].kurtosis()
    skewness = df[col].skew()
    mean = df[col].mean()
    minimum = df[col].min()
    maximum = df[col].max()
    rms_diff = (sum(df[col].diff().fillna(0, inplace=False).apply(lambda x: x*x))/len(df))**0.5
    return standard_devaition, kurtosis, skewness, mean, minimum, maximum, rms_diff

# Get the preprocessed data for feature extraction

def gety_fun(pos, data_frame):
        
    feature = []
    data_frame = data_frame.iloc[pos:][['sao2', 'heartrate', 'respiration', 'cvp', 'systemicsystolic', 'systemicdiastolic', 'systemicmean']]
    
    for col in ['sao2', 'heartrate', 'respiration', 'cvp', 'systemicsystolic', 'systemicdiastolic', 'systemicmean']:
        feature.append(feature_fun(col, data_frame))
    
    return np.array(feature).flatten()

# Settings
time_duration = 10
time_sep = 24
noDataPoints = 24 # 1/5*60*2
time_blocks = np.asarray([0, 1, 2, 3, 4])*noDataPoints
time_prior = 6

# Importing Normalized DF
vital_df=pd.read_csv('vitalPeriodic.csv', chunksize = 10000000)

# Load medication dict
dt = load_obj('med_dt')

# Preprocessing
def process1(demo_df, drugdt, ui):
    demo_df = demo_df[['vitalperiodicid', 'patientunitstayid', 'observationoffset', 'temperature', 'sao2', 'heartrate', 'respiration', 'cvp', 'systemicsystolic', 'systemicdiastolic', 'systemicmean']]
    demo_df[['vitalperiodicid','patientunitstayid', 'observationoffset']] = demo_df[['vitalperiodicid','patientunitstayid', 'observationoffset']].astype('int32')
    demo_df[['temperature','sao2', 'heartrate', 'respiration','cvp', 'systemicsystolic', 'systemicdiastolic', 'systemicmean']] = demo_df[['temperature','sao2', 'heartrate', 'respiration','cvp', 'systemicsystolic', 'systemicdiastolic', 'systemicmean']].astype('float32')
    

    
    
     
    temp = demo_df.dropna(subset=['temperature'])[['vitalperiodicid', 'patientunitstayid', 'observationoffset', 'temperature', 'sao2', 'heartrate', 'respiration', 'cvp', 'systemicsystolic', 'systemicdiastolic', 'systemicmean']]
    
    feverdf = copy.copy(temp)
    feverdf['temperature'] = feverdf['temperature'].apply(lambda x: np.nan if x<38.0 else x)
    feverdf = feverdf.dropna(subset=['temperature'])[['patientunitstayid']]
    
    feverpid = feverdf.patientunitstayid.unique()
    
    pospid = temp.patientunitstayid.unique()
    
    volume = len(temp.patientunitstayid.unique())
    #Removing those pid who does not have any fever data.
    demo_df = demo_df.drop(demo_df[[volume == Sum for Sum in sum([demo_df.patientunitstayid!=pid for pid in temp.patientunitstayid.unique()])]].index)
    
    dt = {}
    colms = ['systemicsystolic', 'systemicdiastolic', 'systemicmean', 'sao2', 'heartrate', 'respiration', 'cvp']
    for pid in demo_df.patientunitstayid.unique():
        for col in colms:
            dt[(pid, col)] = demo_df[demo_df['patientunitstayid']==pid][col].median()
    for pid in temp.patientunitstayid.unique():
        for col in colms:
            temp.loc[(temp.patientunitstayid==pid) & (temp[col].isnull()), col] = dt[(pid, col)]
    temp['systemicdiastolic'] = temp['systemicdiastolic'].fillna(demo_df['systemicdiastolic'].median())
    temp['systemicsystolic'] = temp['systemicsystolic'].fillna(demo_df['systemicsystolic'].median())
    temp['systemicmean'] = temp['systemicmean'].fillna(demo_df['systemicmean'].median())
    temp['sao2'] = temp['sao2'].fillna(demo_df['sao2'].median())
    temp['heartrate'] = temp['heartrate'].fillna(demo_df['heartrate'].median())
    temp['respiration'] = temp['respiration'].fillna(demo_df['respiration'].median())
    temp['cvp'] = temp['cvp'].fillna(demo_df['cvp'].median())
    
    for pid in demo_df.patientunitstayid.unique():
        for col in colms:
            demo_df.loc[(demo_df.patientunitstayid==pid) & (demo_df[col].isnull()), col] = dt[(pid, col)]

    demo_df['systemicdiastolic'] = demo_df['systemicdiastolic'].fillna(demo_df['systemicdiastolic'].median())
    demo_df['systemicsystolic'] = demo_df['systemicsystolic'].fillna(demo_df['systemicsystolic'].median())
    demo_df['systemicmean'] = demo_df['systemicmean'].fillna(demo_df['systemicmean'].median())
    demo_df['sao2'] = demo_df['sao2'].fillna(demo_df['sao2'].median())
    demo_df['heartrate'] = demo_df['heartrate'].fillna(demo_df['heartrate'].median())
    demo_df['respiration'] = demo_df['respiration'].fillna(demo_df['respiration'].median())
    demo_df['cvp'] = demo_df['cvp'].fillna(demo_df['cvp'].median())
    
    col_names =  demo_df.columns
    sorted_df = demo_df.groupby(['patientunitstayid'], as_index=False).apply(pd.DataFrame.sort_values, 'observationoffset').reset_index()
    sorted_df.drop(columns=['level_0', 'level_1'], inplace=True)
    
    df = temp.groupby(['patientunitstayid'], as_index=False).apply(pd.DataFrame.sort_values, 'observationoffset').reset_index()
    df.drop(columns=['level_0', 'level_1'], inplace=True)
    
    
    df_epCount  = pd.DataFrame(columns = ['epCount'])
    for pid in sorted_df.patientunitstayid.unique():
       # print(pid)
        df_fever = sorted_df[sorted_df.patientunitstayid==pid]['temperature'].apply(lambda x: 1 if x>=38.0 else 0)        
        df_epCount = pd.concat([df_epCount, (df_fever.cumsum()-df_fever).apply(lambda x: str(x))])
    
    sorted_df['patientunitstayid'] = sorted_df['patientunitstayid'].apply( lambda x: str(x))
    sorted_df['epCount'] = df_epCount[0]
    sorted_df['patientunitstayid'] =  sorted_df[['patientunitstayid', 'epCount']].apply(lambda x: '_'.join(x), axis=1)
    colm = ['sao2', 'heartrate', 'respiration', 'cvp', 'systemicsystolic', 'systemicdiastolic', 'systemicmean']
    
    
    X=[]
    Y=[]
    Z=[]  
   
        
    temperature_list = sorted_df[sorted_df['temperature'].notnull()].index.tolist()
    i=0
    prev_idx = 0
    pid = sorted_df.iloc[0]['patientunitstayid']
    lastFever = sorted_df.iloc[0]['observationoffset']# does not matter
    prev_pid = -1
    freq = 1/5
    pos = int(60*time_prior*freq)

    for idx in temperature_list: 
        pid = sorted_df.iloc[idx]['patientunitstayid']
        if not pid==prev_pid:
            if not prev_pid==-1:
                lastFever = sorted_df[sorted_df['patientunitstayid'] == prev_pid].iloc[-1]['observationoffset']
            prev_pid = sorted_df.iloc[idx]['patientunitstayid']
            prev_idx = sorted_df.index[sorted_df['patientunitstayid'] == pid].tolist()[0]
            
        if pid.split('_')[0] not in feverpid and sorted_df.iloc[idx]['temperature']<38.0:
            
            if len(sorted_df.loc[prev_idx:idx+1].iloc[:-pos-1])+5<=int(60*time_duration*freq) or drugdt.get(int(pid.split('_')[0]), [0, 0])[0]==1:
                pass
            else:               
                x=[]
                Z.append(pid)
                
                for time_block in time_blocks:                  
                    
                    oop = gety_fun(time_block, sorted_df.loc[prev_idx:idx+1].iloc[:-pos-1])
                
                    x.append(oop)                
                Y.append(0)        
                X.append(np.array(x))
        
        
        elif (not sorted_df.loc[idx]['epCount'] == '0') or drugdt.get(int(pid.split('_')[0]), [0, 0])[1]<sorted_df.iloc[idx]['observationoffset']:        

            if len(sorted_df.loc[prev_idx:idx+1])>5+int(60*time_duration*freq) and (sorted_df.loc[idx]['epCount'] == '0' or sorted_df.loc[idx]['observationoffset'] - lastFever > time_sep*60):                
        
                while len(sorted_df.loc[prev_idx:idx+1])>5+int(60*time_duration*freq):
                    prev_idx+=1

                x=[]
                Z.append(pid)
                
                for time_block in time_blocks:                
                    oop = gety_fun(time_block, sorted_df.loc[prev_idx:idx+1].iloc[:-pos-1])
                    x.append(oop)
                
                Y.append(1)        
                X.append(np.array(x))
                
        i+=1
    
    Y = np.array(Y)
    X = np.array(X)
    np.save('FiverWindowX6hr'+str(ui), X)
    np.save('FiverWindowY6hr'+str(ui), Y)
    np.save('FiverWindowZ6hr'+str(ui), Z)
    np.save('FiverWindowpospid6hr'+str(ui), pospid)
    print('Done for chunk', ui)

    return Y, X, Z, pospid


iu = 0
Y , X, Z = [], [], []
pospid = []
for chunk in vital_df:
    iu+=1
    if iu>0:
        y, x, z, pid= process1(chunk, dt, iu)
        pospid.append(pid)
        print(iu)
        print(len(pid), sum(y), len(y))
        for i in range(len(y)):
            Y.append(y[i])
            X.append(x[i])
            Z.append(z[i])

np.save('FiverWindowX6hr', X)
np.save('FiverWindowY6hr', Y)
np.save('FiverWindowZ6hr', Z)
np.save('FiverWindowpospid6hr', pospid)


