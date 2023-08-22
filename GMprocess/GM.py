#%%
import os
import pandas as pd
from io import StringIO
dir1 = rf"C:../../../brainage/Data"
freesurfer = pd.DataFrame()
for filedir in os.listdir(dir1):
    all_dict = {}
    aseg = os.path.join(dir1, filedir, "stats", "aseg.stats")
    wmparc = os.path.join(dir1, filedir, "stats", "wmparc.stats")
    if os.path.exists(aseg):
        with open(aseg, "r") as f1:
            lines1 = f1.readlines()
            data_start_line1 = 0
            for idx, line in enumerate(lines1):
                if line.startswith("# ColHeaders"):
                    data_start_line1 = idx + 1
                    break

        # aseg_dataform = pd.read_csv(aseg, header=0)
        # wmparc_dataform = pd.read_csv(wmparc, header=0)
        data_text1 = '\n'.join(lines1[data_start_line1:])
        stats_data1 = pd.read_csv(StringIO(data_text1),delim_whitespace=True,header=None)
        with open(wmparc, "r") as f:
            lines = f.readlines()
            data_start_line = 0
            for idx, line in enumerate(lines):
                if line.startswith("# ColHeaders"):
                    data_start_line = idx + 1
                    break

    # aseg_dataform = pd.read_csv(aseg, header=0)
        # wmparc_dataform = pd.read_csv(wmparc, header=0)
        data_text = '\n'.join(lines[data_start_line:])
        stats_data = pd.read_csv(StringIO(data_text),delim_whitespace=True,header=None)
        

        
        # print(stats_data1.iloc[:,0].to_list() + stats_data.iloc[:,0].to_list())
        # all_dict['index'] = [i+1 for i in range(len( stats_data1.iloc[:,3].to_list() + stats_data.iloc[:,3].to_list()))]
        # all_dict['SegId'] = stats_data1.iloc[:,1].to_list() + stats_data.iloc[:,1].to_list()
        all_dict['NVoxels'] = stats_data1.iloc[:,2].to_list() + stats_data.iloc[:,2].to_list()

        all_dict['Volume_mm3'] = stats_data1.iloc[:,3].to_list() + stats_data.iloc[:,3].to_list()
        # all_dict['StructName'] = stats_data1.iloc[:,4].to_list() + stats_data.iloc[:,4].to_list()
        all_dict['normMean'] = stats_data1.iloc[:,5].to_list() + stats_data.iloc[:,5].to_list()
        all_dict['normStdDev'] = stats_data1.iloc[:,6].to_list() + stats_data.iloc[:,6].to_list()
        all_dict['normMin'] = stats_data1.iloc[:,7].to_list() + stats_data.iloc[:,7].to_list()
        all_dict['normMax'] = stats_data1.iloc[:,8].to_list() + stats_data.iloc[:,8].to_list()
        all_dict['normRange'] = stats_data1.iloc[:,9].to_list() + stats_data.iloc[:,9].to_list()
        list_data = []
        for i in range(len(all_dict['NVoxels'])):
            list_data = list_data + pd.DataFrame(all_dict).iloc[i,:].to_list()
        
        
        list_data.append(filedir)
        df = pd.DataFrame(pd.Series(list_data)).T
        freesurfer = pd.concat([freesurfer, df], ignore_index=True)
        
freesurfer.to_csv( "free.csv", index=False)
