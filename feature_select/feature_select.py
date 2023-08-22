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
adni_dataset = pd.read_csv('/home/sdnu/gxl/Brainage/feature/10_29_newpreprocess_adni_nc_pyradiomics.csv')
adni_train_x = adni_dataset.loc[:,adni_dataset.columns.str.startswith('original')]
adni_train_y = adni_dataset['Age'].values.astype(int).tolist()  
Origin_train = adni_train_x.loc[:,adni_train_x.columns.str.endswith('001')]
id = [i for i in range(2,113)] +[i for i in range(115,215)] +[i for i in range(219,247)] 
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
#%%
# for i in range(40,50):
#     model = ['CatBoost']
#     automl_explain = AutoML(mode = 'Optuna',algorithms=model,results_path=f'/home/sdnu/gxl/113andhippmodel/onthers2/{i}',random_state=i)
#     automl_explain.fit(firstorder_Glcm, y)

#%%
model = ['Xgboost']
automl_explain = AutoML(mode = 'Optuna',algorithms=model,results_path=f'/home/sdnu/gxl/113andhippmodel/onthers3',random_state=40)
automl_explain.fit(firstorder_Glcm, y)
# %%
# X = X.filter(regex='^(?!.*shape).*$')
# #%
# X = X.filter(regex='^(?!.*ngtdm).*$')
# %%

#%%
# id = [i for i in range(1,247)] 
# select_all_ROI = pd.DataFrame()
# for index in tqdm(id):
#     threshold = 0.8
#     redundant_features = []
#     index = f"{index:>03}" 
#     select_feature = X.loc[:,X.columns.str.endswith(index)]
#     corr_matrix = select_feature.corr()
#     for i in range(len(corr_matrix.columns)):
#         for j in range(i):
#             if abs(corr_matrix.iloc[i, j]) > threshold:
#                 column = corr_matrix.columns[i]
#                 redundant_features.append(column)
#     select_feature.drop(redundant_features, axis=1, inplace=True)
#     select_all_ROI = pd.concat([select_all_ROI,select_feature],axis=1)
# #%%
# select_all_ROI_col = select_all_ROI.columns.tolist()
# select_all_ROI_col = [s[:-4] for s in select_all_ROI_col]
# counted = Counter(select_all_ROI_col)
# most_common = counted.most_common(30)
# common_words = [i[0] for i in most_common]
# common_words
# all_common_words = []
# id = [i for i in range(1,247)] 
# for i in tqdm(id):
#     i = f"{i:>03}" 
#     all_common_words = all_common_words+[s + f'_{i}' for s in common_words] 
# X =X.filter(all_common_words)
# %%
# model = ['Xgboost']
# automl_explain = AutoML(explain_level=2,algorithms =['Xgboost'],train_ensemble=False,stack_models=False, features_selection=True,validation_strategy={"validation_type": "kfold","k_folds": 10},total_time_limit=10000000000,mode = 'Explain',results_path=f'/home/sdnu/gxl/feature_select/optuna_result/auto_explain_selectbypersonallROI{model}')
# automl_explain.fit(X, y)