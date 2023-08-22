# 绘制草图 草
#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import spearmanr,pearsonr
import numpy as np
from scipy.stats import ttest_ind



# 整理数据
ad_data = pd.read_csv("../../../brainage/Data/data_result/ad_dataM.csv")
mci_data = pd.read_csv("../../../brainage/Data/data_result/mci_dataM.csv")
nc_data = pd.read_csv("../../../brainage/Data/data_result/test_dataM.csv")
EMCI_data = pd.read_csv("../../../brainage/Data/data_result/EMCI_dataM.csv")
LMCI_data = pd.read_csv("../../../brainage/Data/data_result/LMCI_dataM.csv")

ad_data['group'] = "AD"
mci_data['group'] = "MCI"
nc_data['group'] = "NC"
EMCI_data['group'] = "EMCI"
LMCI_data['group'] = "LMCI"
EMCI_data = EMCI_data.rename(columns={'Age_x':'Age'})
LMCI_data = LMCI_data.rename(columns={'Age_x':'Age'})
mci_data = mci_data.rename(columns={'Age_x':'Age'})
ad_data = ad_data.rename(columns={'Age_x':'Age'})
nc_data = nc_data.rename(columns={'Age_x':'Age'})
colname = ['file_name', 'Age', 'predict', 'gap', 'correct_predict',\
         'correct_gap', 'group','MMSE Total Score','Global CDR','FAQ Total Score','NPI-Q Total Score']
ad_data = ad_data.loc[:, colname]
mci_data = mci_data.loc[:, colname]
nc_data = nc_data.loc[:, colname]
EMCI_data = EMCI_data.loc[:, colname]
LMCI_data = LMCI_data.loc[:, colname]



df_plot_gap = pd.concat([ad_data, mci_data, nc_data,EMCI_data,LMCI_data], axis=0)
#%%
nc_data = pd.read_csv("../../../brainage/Data/linchuang/NC.csv")
nc_data.describe()
#%%
df_plot_mmse = pd.read_csv("../../../brainage/Data/data_result/allmmse.csv")
#%%
df_plot_gap.dropna(subset=['MMSE Total Score'], inplace=True)
 #%%
# 可以加散点透明
sns.set_style("whitegrid", {'axes.grid': True})
sns.jointplot(data=df_plot_gap, y="correct_gap", x="MMSE Total Score", kind="reg",color="#4682B4")
plt.scatter(data=df_plot_mmse, y="correct_gap", x="MMSE Total Score", color="#59A95A")
plt.scatter(data=df_plot_mmse[df_plot_mmse['MMSE Total Score']<=28], y="correct_gap", x="MMSE Total Score", c="#F9913D")
plt.scatter(data=df_plot_mmse[df_plot_mmse['MMSE Total Score']<=25], y="correct_gap", x="MMSE Total Score", color="#4D85BD")
plt.xlabel('MMSE')
plt.ylabel("Corrected BAG")

# sns.jointplot(data=df_plot_mmse, y="correct_gap", x="MMSCORE", ax = ax)
# sns.lmplot(data=df_plot_mmse, y="correct_gap", x="MMSCORE", scatter=False, ci=None, fit_reg=True)

plt.show()

print("correct_age_error_corr", spearmanr(df_plot_gap['MMSE Total Score'],df_plot_gap['correct_gap']))
#%%

#
# %%
##绘制箱线图
import seaborn as sns
import matplotlib.pyplot as plt

ad_data['group'] = "AD"
mci_data['group'] = "MCI"
nc_data['group'] = "NC"

colname = ['file_name', 'Age', 'predict', 'gap', 'correct_predict',\
         'correct_gap', 'group']
ad_data = ad_data.loc[:, colname]
mci_data = mci_data.loc[:, colname]
nc_data = nc_data.loc[:, colname]

df_plot = pd.concat([nc_data, mci_data, ad_data], axis=0)
# %%
# 抖动， 透明度，调整z轴，箱子宽度，颜色，边缘线加粗
# fig, ax = plt.subplots(figsize=(4, 3), dpi=80)

colors = {'NC':'#4D85BD','MCI':'#F9913D','AD':'#59A95A'}
sns.stripplot(y="correct_gap", x="group", data=df_plot,palette=colors, alpha=0.6, jitter=0.15, size=6.8, zorder=-1)
# color='#708090'
sns.boxplot(y="correct_gap", x="group", data=df_plot,palette=colors,width=0.3, linewidth=3)
plt.xlabel('Group')
plt.ylabel('Corrected BAG')
plt.show()

# %%
nc_data.describe()
# %%
##绘制correct更改图
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

colors = np.linspace((253, 255, 217), (163, 60, 2), num=36)
colors = tuple(c/255 for c in colors)
cmap = plt.cm.colors.ListedColormap(colors)
fig = plt.figure(figsize=(20, 5))

plt.subplot(1,3,1)

x = sns.kdeplot(x="Age", y="predict", data=nc_data, fill=True, thresh=0.2, n_levels=30, cmap=cmap,)
x.set(xlim=(70, 90), ylim=(70, 90))
plt.grid(linestyle='-', alpha=0.5)
plt.scatter(x="Age", y="predict", data=nc_data, s=5, alpha=1, color='#708090')
plt.xlabel('Chronological Age')
plt.ylabel("Predicted Brain Age ")
# plt.savefig('./3.png')
# plt.close()

plt.subplot(1,3,2)
x2 = sns.kdeplot(x="Age", y="gap", data=nc_data, fill=True, thresh=0.2, n_levels=30, cmap=cmap)
sns.regplot(x="Age", y="gap", data=nc_data, scatter=False,color=(246/255,32/255,32/255),line_kws={"linestyle": "--"})
x2.set(xlim=(70, 90), ylim=(-20, 20))
plt.grid(linestyle='-', alpha=0.5)
plt.scatter(x="Age", y="gap", data=nc_data, s=5, alpha=1,color='#708090')
plt.xlabel('Chronological Age')
plt.ylabel("Brain Age Gape (Corrected)")
# plt.savefig('./4.png')
# plt.close()

plt.subplot(1,3,3)
x3 = sns.kdeplot(x="Age", y="correct_gap", data=nc_data, fill=True, thresh=0.2, n_levels=30,  cmap=cmap)
sns.regplot(x="Age", y="correct_gap", data=nc_data, scatter=False,color=(246/255,32/255,32/255),line_kws={"linestyle": "--"})
x3.set(xlim=(70, 90), ylim=(-20, 20))
plt.grid(linestyle='-', alpha=0.5)
plt.scatter(x="Age", y="correct_gap", data=nc_data, s=5, alpha=1, color='#708090')
plt.xlabel('Chronological Age')
plt.ylabel("Brain Age Gape (Corrected)")
plt.subplots_adjust(wspace=0.5)
plt.show()


# %%
sns.lmplot(data=nc_data, x="Age", y="predict", line_kws={'color':"#4D85BD"}, scatter_kws={'color':"#4D85BD", 'alpha':0.8})
plt.ylabel('Predicted Brain Age')
plt.show()

#%%
sns.lmplot(x="Age", y="gap", data=nc_data, line_kws={'color':"#4D85BD"}, scatter_kws={'color':"#4D85BD", 'alpha':0.8})
plt.ylabel('BAG')
plt.show()
# %%
sns.lmplot(x="Age", y="correct_gap", data=nc_data,  line_kws={'color':"#4D85BD"}, scatter_kws={'color':"#4D85BD", 'alpha':0.8})
plt.ylabel('Corrected BAG')
plt.show()
#%%
##绘制数据分布图
nc = pd.read_csv('../../../brainage/Data/Brainage/feature/10_29_newpreprocess_adni_nc_pyradiomics.csv')
ad_data['group'] = "AD"
mci_data['group'] = "MCI"
nc['group'] = "NC"

colname = ['Age','group']
ad_data = ad_data.loc[:, colname]
mci_data = mci_data.loc[:, colname]
nc = nc.loc[:, colname]
df_plot_gap = pd.concat([nc, mci_data,ad_data ], axis=0)
test_data = pd.read_csv('../../../brainage/Data/correct/test_data.csv')
filtered_df = nc.drop(test_data['Unnamed: 0'])
filtered_df = filtered_df.loc[:,['Age']]
test_data = test_data.loc[:,['Age']]
test_data['group'] = 'Test'
filtered_df['group'] = 'Train'
# %%
df_plot_dataset = pd.concat([filtered_df,test_data], axis=0)

colors = {'NC':'#4D85BD','MCI':'#F9913D','AD':'#59A95A'}
# sns.set_style("whitegrid", {'axes.grid': True})
sns.despine()
plt.subplot(2,2,1)

sns.kdeplot(data=df_plot_gap, x="Age", hue="group", fill=True, common_norm=False, palette=colors,
   alpha=.7,linewidth=1,)


plt.subplot(2,2,2)
sns.histplot(
  data=df_plot_gap, x="Age", hue="group",palette=colors, multiple="stack",
)

plt.subplots_adjust(wspace=0.3,hspace=0.3)
plt.gcf().set_size_inches(8, 5)
plt.show()
#%%
colors = {'Train':'#4D85BD','Test':'#F9913D'}

plt.subplot(2,2,3)
sns.kdeplot(data=df_plot_dataset, x="Age", hue="group", fill=True, common_norm=False, palette=colors,


   alpha=.7,linewidth=1,)


plt.subplot(2,2,4)
sns.histplot(
  data=df_plot_dataset, x="Age", hue="group",palette=colors, multiple="stack",
)

plt.subplots_adjust(wspace=0.3,hspace=0.3)
plt.gcf().set_size_inches(8, 5)
plt.show()
# %%
