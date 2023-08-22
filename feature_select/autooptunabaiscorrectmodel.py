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
adni_dataset = pd.read_csv('/ ../../../brainage/Data/Brainage/feature/10_29_newpreprocess_adni_nc_pyradiomics.csv')
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

#%%
firstorder=X.loc[:,X.columns.str.contains('firstorder')]
# %%
Glcm=X.loc[:,X.columns.str.contains('glcm')]
# %%
firstorder_Glcm =  pd.concat([firstorder,Glcm],axis=1)
#%%
firstorder_Glcm = firstorder_Glcm.reset_index()
firstorder_Glcm = firstorder_Glcm.rename(columns={'index': 'original_index'})
train_x,test_x,train_y,test_y = train_test_split(firstorder_Glcm,y,test_size=0.2,shuffle=True)
test_original_indices = test_x['original_index']
test_x.drop('original_index', axis=1, inplace=True)
train_x.drop('original_index', axis=1, inplace=True)
#%%
model = ['Xgboost']
automl_explain = AutoML(mode = 'Optuna',algorithms=model,results_path=f'/ ../../../brainage/Data/correct/correct_model')
automl_explain.fit(train_x, train_y)
# %%
test_data = adni_dataset.iloc[test_original_indices,:]
test_data= test_data.loc[:,('file_name','Age')]
# %%
test_data.to_csv('/ ../../../brainage/Data/correct/test_data.csv')
test_x.to_csv('/ ../../../brainage/Data/correct/test_radiomcs.csv')


# %%
