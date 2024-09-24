import numpy as np
import os
from pytictoc import TicToc

import pickle
# # Load the model from the file
with open('models/240215_BestBayesianModel.pkl', 'rb') as file:
    loaded_model1 = pickle.load(file)
    
    
# Evaluate on Tune set (not yet used for tuning though for model 1)
from sklearn.model_selection import train_test_split

tune_set=np.load('240215_TuneSet_2012-2013.npy',allow_pickle=True)

#Remove the timestamp & save as float
tune_set=np.concatenate([tune_set[:,:29],tune_set[:,30:]], axis=1).astype('float')

print(tune_set.shape)


#The Labels in the .npy files are like this: 0=Defaulted, 1=Matured (Paid off)
paid_set=tune_set[np.where(tune_set[:,-1]==1)[0]]
default_set=tune_set[np.where(tune_set[:,-1]==0)[0]]

print('Deafult and paid shapes',default_set.shape,paid_set.shape)

X_paid_90, X_paid_10, y_paid_90, y_paid_10 = train_test_split(paid_set[:,:-1]
                                             , paid_set[:,-1]
                                        , test_size=0.1, random_state=42)

del X_paid_90,y_paid_90

#Keeping 10% of The majority (paid) class
paid_set=np.column_stack((X_paid_10,y_paid_10))
tune_set=np.concatenate([paid_set,default_set], axis=0)

print('New train set shape',tune_set.shape)

X_tune=tune_set[:,:-1]
y_tune=tune_set[:,-1]

#Reverse the labels; Now 0=Matured, 1=Deafulted
y_tune=np.array([0 if y==1 else 1 for y in y_tune])

del tune_set,default_set,paid_set


import xgboost as xgb

booster = loaded_model1.get_booster()

# # Get Shapley values using the built-in method of XGBoost
shap_values = booster.predict(xgb.DMatrix(X_tune), pred_contribs=True)

np.save('240220_ShapValuesModel240215.npy',shap_values)
