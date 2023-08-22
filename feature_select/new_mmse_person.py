 
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
 
result = pd.read_csv(r'../../../brainage/Data/correct/correct_model/1_Optuna_Xgboost/predictions_out_of_folds.csv')
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
    print('mae',mean_absolute_error(data['Age'],data['predict']))
    print("rmse",sqrt(mean_squared_error(data['Age'],data['predict'])))
    print('r', pearsonr(data['Age'],data['predict']))
    print('r2', r2_score(data['Age'],data['predict']))
    print("age_error_corr", spearmanr(data['Age'],data['gap']))
    print("mape" , mape(data['Age'],data['predict']))

    print('correct_mae',mean_absolute_error(data['Age'],data['correct_predict']))
    print("correct_rmse",sqrt(mean_squared_error(data['Age'],data['correct_predict'])))
    print('correct_r', pearsonr(data['Age'],data['correct_predict']))
    print('correct_r2', r2_score(data['Age'],data['correct_predict']))
    print("correct_age_error_corr", spearmanr(data['Age'],data['correct_gap']))
    print("correct_mape" , mape(data['Age'],data['correct_predict']))

model = AutoML(results_path='../../../brainage/Data/correct/correct_model')

 
test_data = pd.read_csv('../../../brainage/Data/correct/test_data.csv')
test_x = pd.read_csv('../../../brainage/Data/correct/test_radiomcs.csv')
test_y = test_data['Age'].values
test_predict = model.predict(test_x)
 
a =(test_predict-p1[1])/p1[0]
test_data['predict']= test_predict
test_data['correct_predict'] = a
test_data['correct_gap'] = test_data['correct_predict']-test_data['Age']
test_data['gap'] = test_data['predict'] - test_data['Age']

 
 
##筛选nc含有mmse评分的数据
ncmmse = pd.read_csv('../../../brainage/Data/data_result/NCLC.csv')
ncmmse = ncmmse.rename(columns={'Image ID':'file_name'})

 
# ncmmse = ncmmse[(ncmmse['MMSE Total Score']>=29) & (ncmmse['MMSE Total Score']<=30)]
nc_test_mmse = pd.merge(test_data,ncmmse,on = 'file_name')
nc_test_mmse['group'] = "NC"
 
nc_test_mmse.to_csv('../../../brainage/Data/data_result/test_dataM.csv')
 
pingjia(test_data)
  
ad_dataset = pd.read_csv('../../../brainage/Data/linchuang/finalAD.csv')
ad_x = ad_dataset.iloc[:,ad_dataset.columns.str.startswith('original')]
ad_y = ad_dataset['Age'].values.astype(int).tolist() 
ad_Origin = ad_x.loc[:,ad_x.columns.str.endswith('001')]
id = [i for i in range(2,247)] 
for i in tqdm(id):
    i = f"{i:>03}" 
    ad_select_feature = ad_x.loc[:,ad_x.columns.str.endswith(i)]
    ad_Origin = pd.concat([ad_Origin,ad_select_feature],axis=1)
    ad_test_x = ad_Origin
ad_y = np.array(ad_y)
 
ad_predict = model.predict(ad_test_x)
 
 
ad_data = ad_dataset.loc[:,('new Image ID','Age')]
ad_data['predict'] = ad_predict
ad_data['gap'] = ad_data['predict']-ad_data['Age']
 
ad_correct =(ad_data['predict']-p1[1])/p1[0]
ad_data['correct_predict'] = ad_correct
ad_data['correct']=ad_correct
ad_data['correct_gap']=ad_data['correct']-ad_data['Age']
 
# ad_data.to_csv('../../../brainage/Data/data_result/ad_data.csv')
 
pingjia(ad_data)


 

##筛选AD中包含mmse的书
mmse = pd.read_csv('../../../brainage/Data/data_result/ADLC.csv')
mmse = mmse.rename(columns={'Image ID':'file_name'})
ad_data = ad_data.rename(columns={'new Image ID':'file_name'})
# ad_mmse = mmse[(mmse['MMSCORE']>=4) & (mmse['MMSCORE']<=25)]
ad_data = pd.merge(mmse,ad_data,on='file_name')
 
ad_data['group']='AD'
 
ad_data.to_csv('../../../brainage/Data/data_result/ad_dataM.csv')
 
mci_dataset = pd.read_csv('../../../brainage/Data/linchuang/finalMCI.csv')
mci_x = mci_dataset.iloc[:,mci_dataset.columns.str.startswith('original')]
mci_y = mci_dataset['Age'].values.astype(int).tolist() 
mci_Origin = mci_x.loc[:,mci_x.columns.str.endswith('001')]
id = [i for i in range(2,247)] 
for i in tqdm(id):
    i = f"{i:>03}" 
    mci_select_feature = mci_x.loc[:,mci_x.columns.str.endswith(i)]
    mci_Origin = pd.concat([mci_Origin,mci_select_feature],axis=1)
    mci_test_x = mci_Origin
mci_y = np.array(mci_y)
 
mci_predict = model.predict(mci_test_x)
 
mci_data = mci_dataset.loc[:,('Image ID','Age')]
mci_data['predict'] = mci_predict
mci_data['gap'] = mci_predict - mci_y
 
correct_predict = (mci_predict-p1[1])/p1[0]
mci_data['correct_predict'] = correct_predict
mci_data['correct_gap'] = correct_predict - mci_y
 
# mci_data.to_csv('../../../brainage/Data/data_result/mci_data.csv')
 
pingjia(mci_data)
 
mci_data['Age'].mean()
 
mmse = pd.read_csv('../../../brainage/Data/data_result/MCILC.csv')
 
mmse = mmse.rename(columns={'Image ID':'file_name'})
mci_data = mci_data.rename(columns={'Image ID':'file_name'})
# ad_mmse = mmse[(mmse['MMSCORE']>=4) & (mmse['MMSCORE']<=25)]
 
mci_data = pd.merge(mmse,mci_data,on='file_name')
 
mci_data
 
mci_data['group'] = 'MCI'
 
rest_mci_data =mci_data.loc[:,('correct_gap','MMSE Total Score','GDSCALE Total Score','Global CDR','FAQ Total Score','NPI-Q Total Score','group')]
rest_nc_test_mmse =nc_test_mmse.loc[:,('correct_gap','MMSE Total Score','GDSCALE Total Score','Global CDR','FAQ Total Score','NPI-Q Total Score','group')]
rest_ad_data = ad_data.loc[:,('correct_gap','MMSE Total Score','GDSCALE Total Score','Global CDR','FAQ Total Score','NPI-Q Total Score','group')]
all =   pd.concat([rest_mci_data,rest_nc_test_mmse],axis=0)
all =   pd.concat([all,rest_ad_data],axis=0,ignore_index = True)
 
all.to_csv('../../../brainage/Data/data_result/allceliang.csv')

 
EMCI_data = pd.read_csv('../../../brainage/Data/data_result/EMCI_data.csv')
LMCI_data = pd.read_csv('../../../brainage/Data/data_result/LMCI_data.csv')
 
f_statistic, p_value = f_oneway(EMCI_data['correct_gap'], LMCI_data['correct_gap'],ad_data['correct_gap'],test_data['correct_gap'],mci_data['correct_gap'])    


  
  
EMCI_dataset = pd.read_csv('../../../brainage/Data/linchuang/finalEMCI.csv')
 
EMCI_x = EMCI_dataset.iloc[:,EMCI_dataset.columns.str.startswith('original')]
EMCI_y = EMCI_dataset['Age'].values.astype(int).tolist() 
EMCI_Origin = EMCI_x.loc[:,EMCI_x.columns.str.endswith('001')]
id = [i for i in range(2,247)] 
for i in tqdm(id):
    i = f"{i:>03}" 
    EMCI_select_feature = EMCI_x.loc[:,EMCI_x.columns.str.endswith(i)]
    EMCI_Origin = pd.concat([EMCI_Origin,EMCI_select_feature],axis=1)
    EMCI_test_x = EMCI_Origin
EMCI_y = np.array(EMCI_y)
 
EMCI_predict = model.predict(EMCI_test_x)
 
 
EMCI_data = EMCI_dataset.loc[:,('file_name','Age')]
EMCI_data['predict'] = EMCI_predict
EMCI_data['gap'] = EMCI_data['predict']-EMCI_data['Age']
 
EMCI_correct =(EMCI_data['predict']-p1[1])/p1[0]
EMCI_data['correct_predict'] = EMCI_correct
EMCI_data['correct']=EMCI_correct
EMCI_data['correct_gap']=EMCI_data['correct']-EMCI_data['Age']
 
# EMCI_data.to_csv('../../../brainage/Data/data_result/LMCI_data.csv')
 
pingjia(EMCI_data)
 

 
EMCI_dataset = pd.read_csv('')
mci_data['group'] = 'MCI'
 
filename = EMCI_dataset["file_name"]
 
filename = [i.split('_')[-1].split('.')[0] for i in filename]
 
EMCI_dataset["file_name"] = filename
 
result = ', '.join([string.strip('"') for string in filename])
 
result
 
EMCIlc = pd.read_csv('../../../brainage/Data/linchuang/LMCI.csv')
EMCI_data = pd.read_csv('../../../brainage/Data/data_result/LMCI_data.csv')
 
EMCIlc = EMCIlc.rename(columns={'Image ID':'file_name'})
 

EMCIlc['file_name'] = 'I' + EMCIlc['file_name'].astype(str)
 
finalEMCI =  pd.merge(EMCIlc,EMCI_dataset,on='file_name')

 
finalEMCI.to_csv('../../../brainage/Data/linchuang/finalLMCI.csv')
 
finalEMCI
 
EMCI_dataset
 
EMCI_data = pd.merge(EMCIlc,EMCI_data,on='file_name')
 
EMCI_data.to_csv('../../../brainage/Data/data_result/LMCI_dataM')
 
EMCI_data

