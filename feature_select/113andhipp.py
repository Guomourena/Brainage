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
#%%
adni_dataset = pd.read_csv('/ ../../../brainage/Data/Brainage/feature/10_29_newpreprocess_adni_nc_pyradiomics.csv')
adni_train_x = adni_dataset.loc[:,adni_dataset.columns.str.startswith('original')]
adni_train_y = adni_dataset['Age'].values.astype(int).tolist()  
Origin_train = adni_train_x.loc[:,adni_train_x.columns.str.endswith('215')]
id = [i for i in range(216,219)] + [i for i in range(113,115)]
for i in tqdm(id):
    i = f"{i:>03}" 
    select_feature = adni_train_x.loc[:,adni_train_x.columns.str.endswith(i)]
    Origin_train = pd.concat([Origin_train,select_feature],axis=1)
    X = Origin_train
y = np.array(adni_train_y)
#%%
firstorder=X.loc[:,X.columns.str.contains('firstorder')]
# %%
Glcm=X.loc[:,X.columns.str.contains('glcm')]

# %%
firstorder_Glcm =  pd.concat([firstorder,Glcm],axis=1)
# %%

firstorder_Glcm
#%%
for i in range(40,50):
   
    # automl_explain.fit(Ngtdm, y)
    model = ['Xgboost']
    automl_explain = AutoML(mode = 'Optuna',algorithms=model,results_path=f'/ ../../../brainage/Data/113andhippmodel/hippand113/{i}',random_state=i)
    automl_explain.fit(firstorder_Glcm, y)
#%%
# %%

# %%
