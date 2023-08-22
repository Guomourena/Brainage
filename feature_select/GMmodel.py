
import numpy as np
import nibabel as nib
from skimage.feature import greycomatrix, greycoprops
from sklearn.preprocessing import MinMaxScaler
import ants
import pandas as pd
import os
from tqdm import tqdm
import SimpleITK as sitk
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

#%%
nc = pd.read_csv('../../../brainage/Data/free.csv')
linchuang = pd.read_csv('../../../brainage/Data/linchuang/NC.csv')
# 假设你的DataFrame叫做df，列名为"Column_Name"
linchuang["Image ID"] = linchuang["Image ID"].apply(lambda x: f"I{x}")

nc.rename(columns={'805': 'Image ID'}, inplace=True)
merged_df = pd.merge(nc, linchuang, on='Image ID', how='inner')
adni_train_x = merged_df.iloc[:, 0:805]
adni_train_y = merged_df['Age'].values.astype(int).tolist()  
adni_train_y = np.array(adni_train_y)

for i in range(40,50):
 
    model = ['Xgboost']
    automl_explain = AutoML(mode = 'Optuna',algorithms=model,results_path=f'../../../brainage/Data/experience3/GMmodel/{i}',random_state=i)
    automl_explain.fit(adni_train_x, adni_train_y)
   
