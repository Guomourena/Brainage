#%%
from supervised.automl import AutoML
import shap
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# %%
automl  = AutoML(results_path='/home/sdnu/gxl/feature_select/optuna_result/auto_explain_selectbyfirstorder_GlcmXgboost')
# %%
adni_dataset = pd.read_csv('/home/sdnu/gxl/Brainage/feature/10_29_newpreprocess_adni_nc_pyradiomics.csv')
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
firstorder=X.loc[:,X.columns.str.contains('firstorder')]
Glcm=X.loc[:,X.columns.str.contains('glcm')]
firstorder_Glcm =  pd.concat([firstorder,Glcm],axis=1)
#%%
explainer = shap.Explainer(automl.predict)
# %%
shap_values = explainer.shap_values(firstorder_Glcm)
# %%
