#%%
import pandas as pd
import os

# 设置文件路径
directory = r'../../../brainage/Data/feature_select/optuna_result/auto_explain_selectbyfirstorder_GlcmXgboost/1_Default_Xgboost_SelectedFeatures'
files = [f for f in os.listdir(directory) if f.startswith('learner_fold_') and f.endswith('_shap_importance.csv')]

# 创建一个空的字典用于存储特征的 SHAP 值之和
aggregated_data = {}

# 遍历文件并进行聚合
for filename in files:
    filepath = os.path.join(directory, filename)
    df = pd.read_csv(filepath)
    
    for _, row in df.iterrows():
        feature = row['feature']
        shap_importance = row['shap_importance']
        
        if feature in aggregated_data:
            aggregated_data[feature] += shap_importance
        else:
            aggregated_data[feature] = shap_importance

# 创建一个新的 DataFrame
aggregated_df = pd.DataFrame(list(aggregated_data.items()), columns=['feature', 'shap_importance_sum'])

# 按照 shap_importance_sum 进行降序排序
aggregated_df = aggregated_df.sort_values(by='shap_importance_sum', ascending=False)

# 保存结果到新的 CSV 文件
aggregated_df.to_csv('aggregated_shap_importance.csv', index=False)

# %%

# %%
