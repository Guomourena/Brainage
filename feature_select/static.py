#%%
import pandas as pd

# %%
all = pd.read_csv('/home/sdnu/gxl/linchuang/all.csv')
# %%
MCI_radiomics_feature_train = all[all['file_name'].str.startswith('ADreg_normalize')]
AD_radiomics_feature_train = all[all['file_name'].str.startswith('MCI')]
# %%
MCI_radiomics_feature_train.to_csv('/home/sdnu/gxl/linchuang/MCIradiomics.csv')
# %%
AD_radiomics_feature_train.to_csv('/home/sdnu/gxl/linchuang/ADradiomics.csv')
# %%
ADLC = pd.read_csv('/home/sdnu/gxl/linchuang/AD.csv')
# %%
ADLC
# %%
AD_name = AD_radiomics_feature_train['file_name'].tolist()
# %%
AD_name = [i.split('.')[0].split('_')[3] for i in AD_name]
# %%
AD_radiomics_feature_train['Image ID'] = AD_name


# %%
AD_radiomics_feature_train
# %%
ADLCID = ADLC['Image ID'].tolist()
# %%
modified_list = ["I" + str(word) for word in ADLCID]
# %%
ADLC['Image ID']  = modified_list
# %%
finalAD =  pd.merge(AD_radiomics_feature_train, ADLC, on="Image ID")
# %%
finalAD.to_csv('/home/sdnu/gxl/linchuang/finalAD.csv')
# %%


# %%
