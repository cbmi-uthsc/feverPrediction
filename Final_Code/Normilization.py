import pandas as pd
import numpy as np

# Normlization

vital_df=pd.read_csv('eICU/vitalPeriodic.csv')

for colm in ['sao2', 'heartrate', 'respiration', 'cvp', 'systemicsystolic', 'systemicdiastolic', 'systemicmean']:
    vital_df[colm]=(vital_df[colm]-vital_df[colm].mean())/vital_df[colm].std()
    
vital_df.to_csv("vitalPeriodic.csv")
print("Normalization_Done")
