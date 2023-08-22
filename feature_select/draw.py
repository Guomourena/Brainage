#%%
import numpy as np
import pandas as pd
from nilearn import datasets, plotting
from nilearn.input_data import NiftiMasker
import nibabel as nlb
#%%
# 加载示例数据，包含标准化的MNI152脑图像
brain_img = datasets.load_mni152_template()
mask = nlb.load('/ ../../../brainage/Data/Brainage_3_6/draw/BN_Atlas_246_1mm.nii')

#%%
shap_value = pd.read_csv('/ ../../../brainage/Data/feature_select/shap_draw/marged_shapimportance.csv')
all_shap_value = pd.DataFrame()
id = [i for i in range(1,247)]
all_shap_value['feature'] = id
all_shap_value =pd.merge(all_shap_value, shap_value, on='feature', how='left').fillna(0.005)
all_shap_value.to_csv('/ ../../../brainage/Data/Brainage_3_6/draw/all_shap_value0005.csv',index=False)

nifti_masker = NiftiMasker(mask_img=mask, standardize=True)

affine = mask.affine
voxel_dims = mask.header.get_zooms()[:3]

shap_map = np.zeros(mask.shape)

for i in range(1, 246):

    # Get the voxels corresponding to the current region
    region_voxels = np.argwhere(mask.get_fdata() == i)

    # Calculate the mean Shapley value for the current region
    region_shap = all_shap_value['shap_importance'].iloc[i-1]

    # Fill the Shapley value matrix with the mean value for the current region
    shap_map[region_voxels[:, 0], region_voxels[:, 1], region_voxels[:, 2]] = region_shap

shap_img = nlb.Nifti1Image(shap_map, affine)

nlb.save(shap_img, '/ ../../../brainage/Data/feature_select/shap_draw/shap_map0005.nii.gz')

region_voxels = np.argwhere(mask.get_fdata() == 1)
# %%
shap_value['shap_importance'].iloc[1]

region_shap = shap_value['shap_importance'].iloc[1]

shap_map[region_voxels[:, 0], region_voxels[:, 1], region_voxels[:, 2]] = region_shap

all_shap_value['shap_importance']

tmap_filename = '/ ../../../brainage/Data/feature_select/shap_draw/shap_map.nii.gz'

plotting.plot_stat_map('/ ../../../brainage/Data/Brainage_3_6/draw/shap_map.nii.gz')
plotting.plot_stat_map(tmap_filename, threshold=1)

plotting.plot_glass_brain(tmap_filename, title='plot_glass_brain',
                          )

# 将数组数据映射到大脑图像上，并保存为png格式的文件
plotting.plot_stat_map(tmap_filename, bg_img=brain_img, output_file='random_data.png')

display = plotting.plot_glass_brain(None, display_mode='lzry')
# Here, we project statistical maps with filled=True
display.add_contours(tmap_filename, filled=True)
# and add a title
display.title('Same map but with fillings in the contours')

plotting.plot_glass_brain(
    tmap_filename, threshold=0, colorbar=True, plot_abs=False, display_mode='yz' ,
)
# %%
import matplotlib.pyplot as plt
from matplotlib import colors
from nilearn.plotting import plot_glass_brain


cmap = colors.LinearSegmentedColormap.from_list('my_cmap', [(0, 'white'), (0.5, '#81b6dd'), (1, '#3365a2')])
# %%
display=plotting.plot_glass_brain(tmap_filename, cmap=cmap, vmin=0, vmax=1,display_mode='lzry', colorbar=True)

