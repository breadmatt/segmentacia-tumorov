import os
import SimpleITK as sitk
import glob
import fnmatch

INPUT_DATA = "../input_data/test"

patients = glob.glob(INPUT_DATA + "/*/")
for idx, p in enumerate(patients):
    modalities =[os.path.join(p, f) for f in os.listdir(p)]
    for i, m in enumerate(modalities):
        if "more" in m:
            m = m.replace("\\", "/")
            m = m + "/" + m.split("/")[-1] + ".mha"
            print(m)
            vol = sitk.ReadImage(m, sitk.sitkUInt32)
            vol = sitk.Equal(vol, 0, 1, 0)
            m = m.split(".mha")[0]
            m = m + "_binary.mha"
            vol = sitk.Cast(vol, sitk.sitkUInt32)
            sitk.WriteImage(vol, m)