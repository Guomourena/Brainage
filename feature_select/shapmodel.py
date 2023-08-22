#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn  as sns
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression

from scipy.optimize import curve_fit
from scipy.stats import spearmanr,pearsonr
from supervised.automl import AutoML

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import dump,load

from scipy.optimize import curve_fit
from scipy.stats import spearmanr,pearsonr
from math import sqrt

import shap
# %%
adni_dataset = pd.read_csv('../../../brainage/Data/Brainage/feature/10_29_newpreprocess_adni_nc_pyradiomics.csv')
adni_train_x = adni_dataset.loc[:,adni_dataset.columns.str.startswith('original')]
adni_train_y = adni_dataset['Age'].values.astype(int).tolist()  
Origin_train = adni_train_x.loc[:,adni_train_x.columns.str.endswith('001')]
id = [i for i in range(2,247)] 
for i in tqdm(id):
    i = f"{i:>03}" 
    select_feature = adni_train_x.loc[:,adni_train_x.columns.str.endswith(i)]
    Origin_train = pd.concat([Origin_train,select_feature],axis=1)
    X = Origin_train
y = np.array(adni_train_y)
#%%
firstorder=X.loc[:,X.columns.str.contains('firstorder')]
# %%
# %%
Glcm=X.loc[:,X.columns.str.contains('glcm')]
# # %%
# %%
firstorder_Glcm =  pd.concat([firstorder,Glcm],axis=1)
#%%
automl_explain = AutoML(results_path=f'../../../brainage/Data/feature_select/optuna_result/auto_explain_selectbyfirstorder_GlcmXgboost')
# %%
# %%
