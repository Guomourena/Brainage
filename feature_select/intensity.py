#%%
import pandas as pd
import numpy as np
from joblib import dump
from collections import Counter


from tqdm import tqdm
from sklearn.model_selection import train_test_split,KFold
from supervised.automl import AutoML


from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from scipy.stats import spearmanr,pearsonr
from math import sqrt

from tqdm import tqdm
#%%
intnstiy = pd.read_csv('/home/sdnu/gxl/linchuang/intensity.csv')
intnstiy = pd.read_csv('/home/sdnu/gxl/linchuang/intensity.csv')
intnstiylc= pd.read_csv('/home/sdnu/gxl/linchuang/NC.csv')
intnstiylc = intnstiylc.rename(columns={'Image ID':'File Name'})
intnstiylc['File Name'] = 'I' + intnstiylc['File Name'].astype(str)
finalintnstiy=  pd.merge(intnstiy,intnstiylc,on='File Name')
colname = ['Mean Intensity', 'Std Intensity','Min Intensity', 'Max Intensity']
finalintnstiy_train_x = finalintnstiy.loc[:, colname]
finalintnstiy_train_y = finalintnstiy['Age'].values.astype(int).tolist()  
X = finalintnstiy_train_x
y = np.array(finalintnstiy_train_y)

#%%
# adni_dataset = pd.read_csv('/home/sdnu/gxl/Brainage/feature/10_29_newpreprocess_adni_nc_pyradiomics.csv')
# adni_train_x = adni_dataset.loc[:,adni_dataset.columns.str.startswith('original')]
# adni_train_y = adni_dataset['Age'].values.astype(int).tolist()  
# Origin_train = adni_train_x.loc[:,adni_train_x.columns.str.endswith('215')]
# id = [i for i in range(216,219)]
# for i in tqdm(id):
#     i = f"{i:>03}" 
#     select_feature = adni_train_x.loc[:,adni_train_x.columns.str.endswith(i)]
#     Origin_train = pd.concat([Origin_train,select_feature],axis=1)
#     X = Origin_train
# y = np.array(adni_train_y)
#%%
for i in range(40,50):
    # model = ['Xgboost']
    # automl_explain = AutoML(explain_level=1,algorithms =['Xgboost'],train_ensemble=False,stack_models=False, features_selection=True,validation_strategy={"validation_type": "kfold","k_folds": 10},total_time_limit=10000000000,mode = 'Explain',results_path=f'/home/sdnu/gxl/10times_model/xgboost/')
    # automl_explain.fit(Ngtdm, y)
    model = ['Xgboost']
    automl_explain = AutoML(mode = 'Optuna',algorithms=model,results_path=f'/home/sdnu/gxl/feature_select/intensitymodel/{i}',random_state=i)
    automl_explain.fit(X, y)
# %%
# %%
X
# %%
finalintnstiy_train_x
# %%
