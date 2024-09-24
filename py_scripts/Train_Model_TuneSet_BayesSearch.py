import numpy as np
import os
from pytictoc import TicToc
from datetime import datetime
from sklearn.model_selection import train_test_split

#This dataset still has the timestamp (observation year and month)
#in column 29 (python index) and is dtype=object !
trainset=np.load('240214_TrainSet_1999-2011.npy',allow_pickle=True)

#Remove the timestamp & save as float
trainset=np.concatenate([trainset[:,:29],trainset[:,30:]], axis=1).astype('float')
#Next, we undersample the paid_off class, keeping only 10% of it

#The Labels in the .npy files are like this: 0=Defaulted, 1=Matured (Paid off)
paid_set=trainset[np.where(trainset[:,-1]==1)[0]]
default_set=trainset[np.where(trainset[:,-1]==0)[0]]

print('Deafult and paid shapes',default_set.shape,paid_set.shape)

X_paid_90, X_paid_10, y_paid_90, y_paid_10 = train_test_split(paid_set[:,:-1]
                                             , paid_set[:,-1]
                                        , test_size=0.1, random_state=42)


np.save('240215_Model_TrainPaid90_set.npy', np.column_stack((X_paid_90,y_paid_90)))

del X_paid_90,y_paid_90

#Keeping 10% of The majority (paid) class
paid_set=np.column_stack((X_paid_10,y_paid_10))
trainset=np.concatenate([paid_set,default_set], axis=0)

np.save('240215_Model_TrainUndersampledPaid10_trainset.npy', trainset)

print('New train set shape',trainset.shape)

X_train=trainset[:,:-1]
y_train=trainset[:,-1]

#Reverse the labels; Now 0=Matured, 1=Deafulted
y_train=np.array([0 if y==1 else 1 for y in y_train])

del trainset,default_set,paid_set

#Get the Tune Set
#This dataset still has the timestamp (observation year and month)
#in column 28 (python index) and is dtype=object !
tuneset=np.load('240215_TuneSet_2012-2013.npy',allow_pickle=True)

#Remove the timestamp & save as float
tuneset=np.concatenate([tuneset[:,:29],tuneset[:,30:]], axis=1).astype('float')

#Next, we undersample the paid_off class, keeping only 10% of it

#The Labels in the .npy files are like this: 0=Defaulted, 1=Matured (Paid off)
paid_set=tuneset[np.where(tuneset[:,-1]==1)[0]]
default_set=tuneset[np.where(tuneset[:,-1]==0)[0]]

print('Deafult and paid shapes',default_set.shape,paid_set.shape)

X_paid_90, X_paid_10, y_paid_90, y_paid_10 = train_test_split(paid_set[:,:-1]
                                             , paid_set[:,-1]
                                        , test_size=0.1, random_state=42)


#np.save('240113_Model3_TunePaid90_set.npy', np.column_stack((X_paid_90,y_paid_90)))

del X_paid_90,y_paid_90

#Keeping 10% of The majority (paid) class
paid_set=np.column_stack((X_paid_10,y_paid_10))
tuneset=np.concatenate([paid_set,default_set], axis=0)

#np.save('240113_Model3_TuneUndersampledPaid10_trainset.npy', tuneset)

print('New tune set shape',tuneset.shape)

X_tuning=tuneset[:,:-1]
y_tuning=tuneset[:,-1]

#Reverse the labels; Now 0=Matured, 1=Deafulted
y_tuning=np.array([0 if y==1 else 1 for y in y_tuning])

del tuneset,default_set,paid_set


#Model initialize and train

import xgboost as xgb
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization


# Define a range of hyperparameters to search
param_bounds = {
    'n_estimators': (500, 1000),  # Number of trees
    'learning_rate': (0.01, 0.1),  # Learning rate
    'max_depth': (2, 18),  # Maximum depth of trees
#     'min_child_weight': np.arange(1, 10),  # Minimum sum of instance weight in a child, default=1, found previously 1
    'gamma': (0., 5.),  # Regularization term for tree split
    'subsample': (0.45, 1.0),  # Fraction of samples used for fitting trees
    'colsample_bytree': (0.45, 1.0),  # Fraction of features for tree construction
    'reg_alpha': (0., 5.),  # L1 regularization term
    'reg_lambda': (0., 5.),  # L2 regularization term
#     'scale_pos_weight': [list(y_train).count(0)/list(y_train).count(1)] # DO NOT REBALANCE
#    'max_delta_step':(0, 11)
}

# Log file for tracking hyperparameters
log_file = open(f'240215_Model_TuneSet_hyperparameter_log_{datetime.now().strftime("%Y%m%d%H%M%S")}.txt', 'w')


# Define the objective function for Bayesian optimization
def objective(n_estimators,learning_rate, max_depth,gamma,subsample,colsample_bytree,reg_alpha,reg_lambda):#,max_delta_step):
    model = xgb.XGBClassifier(
        n_estimators=int(n_estimators),
        learning_rate=learning_rate,
        max_depth=int(max_depth),
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        #max_delta_step=max_delta_step,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)  # Train on the entire historical dataset

    # Evaluate on the tuning set and calculate ROC AUC
    y_prob = model.predict_proba(X_tuning)[:, 1]
    roc_auc = roc_auc_score(y_tuning, y_prob)
    
    
    # Log hyperparameters and performance
    log_entry = f"ROC AUC: {roc_auc:.6f}, N Estimators: {n_estimators:.0f}, LR: {learning_rate:.6f}, Max Depth: {max_depth:.0f}, Gamma: {gamma:.3f}, Subsample: {subsample:.3f}, " \
                f"Colsample Bytree: {colsample_bytree:.3f}, reg_alpha: {reg_alpha:0.3f}, reg_lambda: {reg_lambda:0.3f} \n"
    log_file.write(log_entry)
    log_file.flush()
    

    # Return the negative ROC AUC (as Bayesian optimization aims to maximize)
    return roc_auc


# Perform Bayesian optimization
optimizer = BayesianOptimization(
    f=objective,
    pbounds=param_bounds,
    random_state=42
)

t = TicToc() ## TicToc("name")

print('Start hyper parameter search')
t.tic()

optimizer.maximize(
    init_points=350,  # Number of random points to explore
    n_iter=50       # Number of optimization steps
    )

t.toc()


# Extract the best hyperparameters
best_params = optimizer.max['params']



# Train the final model on the entire historical training set using the best hyperparameters
final_model = xgb.XGBClassifier(
        n_estimators=int(best_params['n_estimators']),
        learning_rate=best_params['learning_rate'],
        max_depth=int(best_params['max_depth']),
        gamma=best_params['gamma'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        reg_alpha=best_params['reg_alpha'],
        reg_lambda=best_params['reg_lambda'],
        #max_delta_step=best_params['max_delta_step'],
        random_state=42,
        n_jobs=-1,
)

final_model.fit(X_train, y_train)


import pickle
# Get the best estimator (trained model with the best hyperparameters)
# Save the best model to a file using pickle
with open('240215_BestBayesianModel.pkl', 'wb') as file:
    pickle.dump(final_model, file)
