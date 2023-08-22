
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn  as sns

from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression

from scipy.optimize import curve_fit
from scipy.stats import spearmanr,pearsonr


import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline



from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from supervised.automl import AutoML

from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error

from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

from joblib import dump,load

from scipy.optimize import curve_fit
from scipy.stats import spearmanr,pearsonr
from math import sqrt
#%%

ixi_dataset = pd.read_csv('/home/sdnu/gxl/Brainage/feature/10_30_new_ixi_pyradiomics.csv')
ixi_dataset.dropna(subset=['AGE'],axis=0,inplace=True)
ixi_dataset =  ixi_dataset.loc[(ixi_dataset['AGE']>=70)]
ixi_train_x = ixi_dataset.loc[:,ixi_dataset.columns.str.startswith('original')]
ixi_train_y = ixi_dataset['AGE'].values.astype(int).tolist()   

# %%

automl_explain = AutoML(results_path=f'/home/sdnu/gxl/feature_select/optuna_result/auto_optuna[\'Xgboost\']')

# %%
predict = automl_explain.predict(ixi_train_x)
# %%
mean_absolute_error(predict,ixi_train_y)
# %%
ixi_train_y
# %%
