# %%
import pandas as pd

import os


train_dirpath = "/../../../brainage/Data/NC"
test_dirpath = "/../../../brainage/Data/MCI"

paths = []

train_ad_dir = os.path.join(train_dirpath, "MCI")
train_nc_dir = os.path.join(train_dirpath, "NC")
for path in os.listdir(train_ad_dir):
    paths.append(
        (os.path.join(train_ad_dir,path), "MCI")
    )

for path in os.listdir(train_nc_dir):
    paths.append(
        (os.path.join(train_nc_dir,path), "NC")
    )


print(len(paths))
# %%
paths
#%%

# test_paths = []
# test_dirpath = "../../../xfdata/Test"
# for path in os.listdir(test_dirpath):
#     test_paths.append(
#         (os.path.join(test_dirpath,path), "unkown")
#     )
# print(test_paths)
# df_test = pd.DataFrame(test_paths, columns=['path', 'label'])

# df_test.to_csv("../../../user_data/test.csv", index=False)
# %%
# %%
train_paths = pd.DataFrame(paths, columns=['path', "label"])
train_paths.to_csv("../../../user_data/train.csv", index=False)
# %%
# test_paths.extend(paths)
# df_all = pd.DataFrame(test_paths, columns=['path', 'label'])

# df_all.to_csv("../../../user_data/all.csv", index=False)


# %%

# %%
