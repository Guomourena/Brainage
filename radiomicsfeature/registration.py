# %%
import ants
import numpy as np
import os
import pandas as pd

def registration(image_path, target_path, output_path):
    source_img = ants.image_read(image_path)
    target_image = ants.image_read(target_path)
    # source_img_nib = nib.load(os.path.join(Image_path, Image_list[i]))
    source_img = ants.denoise_image(source_img, ants.get_mask(source_img))
    source_img_n4 = ants.n4_bias_field_correction(source_img)
    output = ants.registration(target_image, source_img_n4, type_of_transform='SyN')['warpedmovout']
    image_data = output.numpy()
    image_data_flatten = np.sort(image_data.flatten())
    max_95 = int(len(image_data_flatten) * 0.95)
    data_255 = image_data / image_data_flatten[max_95] * 255


    output_img = ants.from_numpy(data_255, target_image.origin, target_image.spacing, target_image.direction)
    image_name = os.path.split(image_path)[1]
    if image_name.endswith(".nii.gz"):
        output_file =  'normalize_255_' + image_name
    else:
        output_file = 'normalize_255_' + image_name + '.gz'
        # output_file = 'normalize_255_' + image_name  
    output_img.to_file(os.path.join(output_path, output_file))

    # return output_file


# %%


#%%
def difregistration(nii_paths,LR,template_path,output_path):
    nii_csv = pd.read_csv(nii_paths)
    LR_list = nii_csv[nii_csv['label']==LR]
    LR_path_list = LR_list['path'].to_list()
    for path in LR_path_list:
        print(path)
        registration(path, template_path, output_path)
   
template_path = "../../../../brainage/Data/MNI152_T1_1mm.nii"
NC_output_path = '../../../../brainage/Data/normalize_data/NC'
# test_output_path = '../../../../brainage/Data/Train/normalize_data/test'
AD_output_path = '../../../user_data/normalize_data/MCI'
all_path = "/../../../brainage/Data/all.csv"
NC = 'NC'
difregistration(all_path,NC,template_path,NC_output_path)    
AD = 'MCI'
difregistration(all_path,AD,template_path,AD_output_path)   
#%% 
