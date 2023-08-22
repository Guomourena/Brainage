#%%

import os

import SimpleITK as sitk
import pandas as pd

from tqdm import tqdm
from radiomics import featureextractor


# 构造特征提取器
settings = {}
settings['binWidth'] = 25
settings['resampledPixelSpacing'] = None  # [3,3,3]
settings['interpolator'] = sitk.sitkBSpline
settings['enableCExtensions'] = True

extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

                                                                                                         
# 全部的mask



masks_id = [i for i in range(1,247)]

for mask_id in tqdm(masks_id):
    print(f"extracting features from ROI {mask_id}.")


    mask_path = f"../../../brainage/Data/MASK/mask_region_{mask_id}.nii.gz"
    print(mask_path)
    image_train_MCI_list = [os.path.join("../../../brainage/Data/normalize_data/MCI/", p) \
        for p in os.listdir("../../../brainage/Data/normalize_data/MCI/")]
    image_train_nc_list = [os.path.join("../../../brainage/Data/normalize_data/NC/", p) \
        for p in os.listdir("../../../brainage/Data/normalize_data/NC/")]
    image_test_list = [os.path.join("../../../brainage/Data/normalize_data/test/", p) \
        for p in os.listdir("../../../brainage/Data/normalize_data/test/")]


    image_train_list = image_train_MCI_list + image_train_nc_list + image_test_list
    df = pd.DataFrame()
    for image_path in image_train_list:
        print(image_path)
        # 特征提取
        featureVector = extractor.execute(image_path,mask_path)
        # print(os.path.exists(image_path))
        df_add = pd.DataFrame.from_dict(featureVector.values()).T
        df_add.columns = featureVector.keys()
        file_name = image_path.split("data/")[-1].replace("/","_")
        df_add.insert(0, 'file_name', file_name)
        df = pd.concat([df,df_add])
    df.to_csv(f'../../../brainage/Data/refeature1/{mask_id:>03}.csv', index=False)

#%%

# %%
