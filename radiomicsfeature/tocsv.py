#%%

import pandas as pd


# %%
def drop_diagnostics_columns(df, mask_id):

    select_colummns = ['file_name']
    select_colummns.extend([ i for i in df.columns if i.startswith("original")])
    df = df[select_colummns]
    re_columns = [i+f"_{mask_id}" for i in df]
    df.columns = re_columns
    df.rename(columns={f"file_name_{mask_id}": "file_name"}, inplace=True)
    return df

# %%
# %%

f = open('../../../brainage/Data/picked_features.txt', 'r')
picked_feature = eval(f.read())
df_all = pd.read_csv(".../../../brainage/Data/re_feature/001.csv",sep=',',usecols=picked_feature)
df_all = drop_diagnostics_columns(df_all, "001")

#%%
for i in range(2, 247):
    path = f'.../../../brainage/Data/re_feature/{i:>03}.csv'
    df_roi = pd.read_csv(path,sep=',',usecols=picked_feature)
    df_roi = drop_diagnostics_columns(df_roi,f"{i:>03}")
    df_all = pd.merge(df_all, df_roi, on="file_name")
#%%
df_all
#%%
df_all.sort_values(by="file_name", inplace=True)
df_all.to_csv(".../../../brainage/Data/selected_radiomics_feature_raw1.csv", index=False)

#%%
# 删除相同列后的特征数据
df_all_remove = df_all.loc[:, (df_all != df_all.iloc[0]).any()]
df_all_remove.to_csv(".../../../brainage/Data/selected_radiomics_feature1.csv",index=False)

#%%
df_all_unselect = pd.read_csv(".../../../brainage/Data/re_feature/001.csv")
df_all_unselect = drop_diagnostics_columns(df_all_unselect, "001")
#%%
for i in range(2, 247):
    path = f'.../../../brainage/Data/re_feature/{i:>03}.csv'
    df_roi = pd.read_csv(path)
    df_roi = drop_diagnostics_columns(df_roi,f"{i:>03}")
    df_all = pd.merge(df_all, df_roi, on="file_name")
df_all.sort_values(by="file_name", inplace=True)
df_all.to_csv(".../../../brainage/Data/re_radiomics_feature_raw1.csv", index=False)
df_all_remove = df_all.loc[:, (df_all != df_all.iloc[0]).any()]
df_all_remove.to_csv(".../../../brainage/Data/re_radiomics_feature1.csv",index=False)
#%%
#拆分测试集和训练集
radiomics_feature = pd.read_csv('.../../../brainage/Data/selected_radiomics_feature1.csv')
# radiomics_feature.sort_values(by= 'file_name',inplace=True)
MCI_radiomics_feature_train = radiomics_feature[radiomics_feature['file_name'].str.startswith('MCI')]
NC_radiomics_feature_train = radiomics_feature[radiomics_feature['file_name'].str.startswith('NC')]
radiomics_feature_train = MCI_radiomics_feature_train.append(NC_radiomics_feature_train)
radiomics_feature_train.to_csv(".../../../brainage/Data/selected_train_radiomics_all1.csv", index=False)
radiomics_feature_test = radiomics_feature[radiomics_feature['file_name'].str.startswith("test")]
radiomics_feature_test.to_csv(".../../../brainage/Data/selected_test_radiomics_all1.csv",index=False)
#%%
radiomics_feature = pd.read_csv('.../../../brainage/Data/re_radiomics_feature1.csv')
# radiomics_feature.sort_values(by= 'file_name',inplace=True)
MCI_radiomics_feature_train = radiomics_feature[radiomics_feature['file_name'].str.startswith('MCI')]
NC_radiomics_feature_train = radiomics_feature[radiomics_feature['file_name'].str.startswith('NC')]
radiomics_feature_train = MCI_radiomics_feature_train.append(NC_radiomics_feature_train)
radiomics_feature_train.to_csv(".../../../brainage/Data/re_train_radiomics_all1.csv", index=False)
radiomics_feature_test = radiomics_feature[radiomics_feature['file_name'].str.startswith("test")]
radiomics_feature_test.to_csv(".../../../brainage/Data/re_test_radiomics_all1.csv",index=False)
# %%
