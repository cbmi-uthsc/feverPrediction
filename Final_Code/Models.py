import pandas as pd
import numpy as np
import random as rnd
import os
import copy
# visualization


# Preprocessing
from keras import backend as K
from sklearn import preprocessing
import datetime

# Deep learning
from keras.layers import Dense
from keras.models import Input, Model
from tcn import TCN

# Machine Learning
from sklearn.linear_model import LogisticRegression
import xgboost as clf1
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier


# Loading
X_ = np.concatenate(tuple([np.load('FiverWindowX6hr'+str(i+1)+'.npy') for i in range(121)]))  #Chunk 121 did'nt had any temperature data so we excluded that
_X = np.concatenate(tuple([np.load('FiverWindowX6hr'+str(i+123)+'.npy') for i in range(25)]))
X = np.concatenate((X_, _X))

Y_ = np.concatenate(tuple([np.load('FiverWindowY6hr'+str(i+1)+'.npy') for i in range(121)]))
_Y = np.concatenate(tuple([np.load('FiverWindowY6hr'+str(i+123)+'.npy') for i in range(25)]))
Y = np.concatenate((Y_, _Y))

# TCNs
# 1 Downsampling data
_X = []
_Y = []
for i in range(len(Y)):
    if Y[i]==1:
        _X.append(X[i])
        _Y.append([0,1])
    elif np.random.rand()<0.2:
        _X.append(X[i])
        _Y.append([1,0])
_X = np.asarray(_X)

# F1 Score
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    print(precision, recall, 2*((precision*recall)/(precision+recall+K.epsilon())))
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# The TCN model

batch_size, timesteps, input_dim = None, None, 49

i = Input(batch_shape=(batch_size, timesteps, input_dim))

o = TCN(kernel_size=2, nb_stacks=2, dilations=[1, 2, 4, 8], return_sequences=False)(i)  # The TCN layers are here.
o = Dense(32, activation='relu')(o)
o = Dense(2, activation='softmax')(o)
m = Model(inputs=[i], outputs=[o])
m.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1])

_Y = np.array(_Y)
_X = np.array(_X)
temp, X_cv, label, y_cv = train_test_split(_X, _Y, test_size=0.2, random_state=2)

m.fit(temp, label, epochs=10, validation_data=(X_cv, y_cv))

# TCN result
print('TCN result')
print(classification_report(np.argmax(y_cv, 1), np.argmax(m.predict(X_cv), 1)))
print('F1 score:', f1_score(np.argmax(y_cv, 1), np.argmax(m.predict(X_cv), 1)))
print('AUC_ROC:', roc_auc_score(np.argmax(y_cv, 1), np.argmax(m.predict(X_cv), 1)))

# Machine Learning

# Some preprocessing
X = np.reshape(X, (X.shape[0], 5*49))
df = pd.DataFrame(data=X)
df['label'] = Y
cases_df = df[df['label']==1]
controls_df = df[df['label']==0]
labels = cases_df['label']
cases_df.drop(columns=['label'], inplace = True)
fever_X_train, fever_x_cv, fever_label, fever_y_cv = train_test_split(cases_df, labels, test_size=0.2, random_state=4)
labels = controls_df['label']
controls_df.drop(columns=['label'], inplace = True)
X_train, x_cv, label, y_cv = train_test_split(controls_df, labels, test_size=0.2, random_state=4)

# Random Forest
clf = RandomForestClassifier(n_estimators=2)
clf = clf.fit(pd.concat([fever_X_train, X_train]), pd.concat([fever_label, label]))

print('Random Forest result')
print(classification_report(pd.concat([fever_y_cv, y_cv]), clf.predict(pd.concat([fever_x_cv, x_cv]))))
print('F1 score:', f1_score(clf.predict(pd.concat([fever_x_cv, x_cv])), pd.concat([fever_y_cv, y_cv])))
print('AUC_ROC:', roc_auc_score(pd.concat([fever_y_cv, y_cv]), clf.predict(pd.concat([fever_x_cv, x_cv]))))

# LogisticRegression
clf = LogisticRegression()
clf = clf.fit(pd.concat([fever_X_train, X_train]), pd.concat([fever_label, label]))
print('LogisticRegression result')
print(classification_report(pd.concat([fever_y_cv, y_cv]), clf.predict(pd.concat([fever_x_cv, x_cv]))))
print('F1 score:', f1_score(clf.predict(pd.concat([fever_x_cv, x_cv])), pd.concat([fever_y_cv, y_cv])))

# GaussianNB
clf = GaussianNB()
clf = clf.fit(pd.concat([fever_X_train, X_train]), pd.concat([fever_label, label]))
print('GaussianNB result')
print(classification_report(pd.concat([fever_y_cv, y_cv]), clf.predict(pd.concat([fever_x_cv, x_cv]))))
print('F1 score:', f1_score(clf.predict(pd.concat([fever_x_cv, x_cv])), pd.concat([fever_y_cv, y_cv])))
print('AUC_ROC:', roc_auc_score(pd.concat([fever_y_cv, y_cv]), clf.predict(pd.concat([fever_x_cv, x_cv]))))
print('AUC_ROC:', roc_auc_score(pd.concat([fever_y_cv, y_cv]), clf.predict(pd.concat([fever_x_cv, x_cv]))))


# Xgboost

# Some preprocessing
df = pd.DataFrame(data=X)
df['label'] = Y
cases_df = df[df['label']==1]
controls_df = df[df['label']==0]
labels = controls_df['label']
X_train, x_cv, label, y_cv = train_test_split(controls_df, labels, test_size=0.2, random_state=23)
labels = cases_df['label']
fever_X_train, fever_x_cv, fever_label, fever_y_cv = train_test_split(cases_df, labels, test_size=0.2, random_state=23)

def get_controls(df):
    downsampled_df, _, _, _ = train_test_split(df, df['label'], test_size=0.01)
    return downsampled_df

# hyperameter
params = {'eta': 0.7, 'max_depth': 6, 'scale_pos_weight': 1, 'objective': 'reg:linear','subsample':0.25,'verbose': False}
xgb_model = None

print('Xgboost model')

runs = 10
Temp_X_cv = copy.copy(fever_x_cv)
Temp_y_cv = copy.copy(fever_y_cv)
for i in range(runs):
    
    pf = pd.concat([fever_X_train, get_controls(X_train).reset_index(drop=True)])
    labels = pf['label']
  #  train_df = pf.drop(['temperature'],axis=1)
   
    print("count: ", i+1)
    print(sum(labels), len(labels))
    if True:
      #  xgb_model = 'model.model'
        temp, X_cv, label, Y_cv = train_test_split(pf, labels, test_size=0.05)
        xg_train_1 = clf1.DMatrix(temp.drop(['label'],axis=1), label=label)
        xg_test = clf1.DMatrix(X_cv.drop(['label'],axis=1), label=Y_cv)
        model = clf1.train(params, xg_train_1, 50, xgb_model=xgb_model)
        model.save_model('model.model')
        xgb_model = 'model.model'
    
    
   # print(clf1.score(X_cv.drop(['temperature'],axis=1), y_cv))
      
        
        print("2nd cv score", i)

        CV_X = pd.concat([fever_x_cv, x_cv])
        cv_y = pd.concat([fever_y_cv, y_cv])
        print(classification_report(cv_y, (model.predict(clf1.DMatrix(CV_X.drop(['label'],axis=1), label=cv_y))>0.5).astype(int)))
        print('F1 score:', f1_score(cv_y, (model.predict(clf1.DMatrix(CV_X.drop(['label'],axis=1), label=cv_y))>0.5).astype(int)))

    
        Temp_X_cv = pd.concat([Temp_X_cv, X_cv])
        Temp_y_cv = pd.concat([Temp_y_cv, Y_cv])

    
#print(clf1.score(Temp_X_cv.drop(['temperature'],axis=1), Temp_y_cv))

print("1st cv score")
print(classification_report(Temp_y_cv, (model.predict(clf1.DMatrix(Temp_X_cv.drop(['label'],axis=1), label=Temp_y_cv))>0.5).astype(int)))
print('F1 score:', f1_score(Temp_y_cv, (model.predict(clf1.DMatrix(Temp_X_cv.drop(['label'],axis=1), label=Temp_y_cv))>0.5).astype(int)))

print("2nd cv final real cv score")

CV_X = pd.concat([fever_x_cv, x_cv])
cv_y = pd.concat([fever_y_cv, y_cv])
print(classification_report(cv_y, (model.predict(clf1.DMatrix(CV_X.drop(['label'],axis=1), label=cv_y))>0.5).astype(int)))
print('F1 score:', f1_score(cv_y, (model.predict(clf1.DMatrix(CV_X.drop(['label'],axis=1), label=cv_y))>0.5).astype(int)))
print('AUC_ROC:', roc_auc_score(cv_y, (model.predict(clf1.DMatrix(CV_X.drop(['label'],axis=1), label=cv_y))>0.5).astype(int)))



