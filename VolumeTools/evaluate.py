import SimpleITK as sitk
import numpy as np
import glob
from PIL import Image
import os
TEST_DATA = "../sliced_images/test"
OUTPUT_DIR = ""
PREDICTION_DIR = "Predictions"


def get_sitk_image(path):
    files = glob.glob(path)
    images = []
    for idx, file in enumerate(files):
        image = np.array(Image.open(file))
        images.append(image)
    volume = np.array(images)
    return sitk.GetImageFromArray(volume)

patients = os.listdir(TEST_DATA)
dice_list = []
for idx, p in enumerate(patients):
    p_dir = os.path.join(TEST_DATA, p)
    if os.path.isdir(os.path.join(p_dir, PREDICTION_DIR)):

        predictions_path = os.path.join(p_dir, PREDICTION_DIR) + "/*.bmp"
        labels_path = os.path.join(p_dir, "Labels") + "/*.bmp"

        pred_volume = get_sitk_image(predictions_path)
        gt_volume = get_sitk_image(labels_path)

        o_metric = sitk.LabelOverlapMeasuresImageFilter()
        o_metric.Execute(pred_volume, gt_volume)
        print(o_metric.GetDiceCoefficient())
        dice_list.append(o_metric.GetDiceCoefficient())

        pred_vol_dir = os.path.join(p_dir, "PredictedVolume")

        if not os.path.exists(pred_vol_dir):
            os.makedirs(pred_vol_dir)

        pred_volume = sitk.Equal(pred_volume, 0, 1, 0);
        pred_volume = sitk.Cast(pred_volume, sitk.sitkUInt32)
        sitk.WriteImage(pred_volume, os.path.join(pred_vol_dir, "volume.mha"))
print("mean", np.mean(np.array(dice_list)))
print("std", np.std(np.array(dice_list)))