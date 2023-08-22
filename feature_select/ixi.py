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

from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# %%
result = pd.read_csv(r'/home/sdnu/gxl/correct/correct_model/1_Optuna_Xgboost/predictions_out_of_folds.csv')
##偏差矫正
def remove_covariates(x,a,b):
    return a*x.iloc[:,0]+b
##回归计算
p1 ,p2 = curve_fit(remove_covariates,result,result.iloc[:,1])

##MAPE评价标准
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) 
##
def pingjia(data):
    print('mae',mean_absolute_error(data['AGE'],data['predict']))
    print("rmse",sqrt(mean_squared_error(data['AGE'],data['predict'])))
    print('r', pearsonr(data['AGE'],data['predict']))
    print('r2', r2_score(data['AGE'],data['predict']))
    print("age_error_corr", spearmanr(data['AGE'],data['gap']))
    print("mape" , mape(data['AGE'],data['predict']))

    print('correct_mae',mean_absolute_error(data['AGE'],data['correct_predict']))
    print("correct_rmse",sqrt(mean_squared_error(data['AGE'],data['correct_predict'])))
    print('correct_r', pearsonr(data['AGE'],data['correct_predict']))
    print('correct_r2', r2_score(data['AGE'],data['correct_predict']))
    print("correct_age_error_corr", spearmanr(data['AGE'],data['correct_gap']))
    print("correct_mape" , mape(data['AGE'],data['correct_predict']))
#%%
model = AutoML(results_path='/home/sdnu/gxl/correct/correct_model')
ixi_dataset = pd.read_csv('/home/sdnu/gxl/Brainage/feature/10_30_new_ixi_pyradiomics.csv')
ixi_dataset.dropna(subset=['AGE'],axis=0,inplace=True)
ixi_dataset =  ixi_dataset.loc[(ixi_dataset['AGE']>=70)]
ixi_train_x = ixi_dataset.loc[:,ixi_dataset.columns.str.startswith('original')]
ixi_train_y = ixi_dataset['AGE'].values.astype(int).tolist()   
ixi_predict = model.predict(ixi_train_x)
ixi_correct =(ixi_predict-p1[1])/p1[0]
ixi_data=ixi_dataset.loc[:,('file_name','AGE')]
ixi_data['predict'] = ixi_predict
gap = ixi_data['predict']-ixi_data['AGE']
ixi_data['gap'] = gap
ixi_correct =(ixi_data['predict']-p1[1])/p1[0]
ixi_data['correct_predict'] = ixi_correct
ixi_data['correct_gap'] =ixi_data['correct_predict']-ixi_data['AGE']
ixi_data.to_csv('/home/sdnu/gxl/data_result/ixi_data.csv')
print(pingjia(ixi_data))
# %%
ixi_data
# %%
