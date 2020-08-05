from utils import *

def transform_volumes_into_slice_images(sample_dir_names, volume_paths, labels_paths, output_dir, mri_sequence_flag, labels_flag):
    for idx, patient in enumerate(sample_dir_names):
        print(f"{idx + 1}/{len(volume_paths)}")
        path = os.path.join(output_dir, patient)
        if not os.path.exists(path):
            os.makedirs(path)
        sequence_path = os.path.join(path, mri_sequence_flag)
        #if not os.path.exists(sequence_path):
        #    os.makedirs(sequence_path)
        labels_path = os.path.join(path, labels_flag)
        #if not os.path.exists(labels_path):
        #    os.makedirs(labels_path)
        volume = load_volume(volume_paths[idx])
        #normalized0_1 = normalize0_1(sitk.GetArrayFromImage(volume))
        normalized0_255 = normalize0_255(sitk.GetArrayFromImage(volume))


        labels = load_volume(labels_paths[idx], isLabels=True)
        binarized_labels = binarize_labels(labels)

        #save_slices_as_bmp(normalized0_1, sequence_path + "/")
        save_slices_as_bmp(normalized0_255, sequence_path + "/")

        save_slices_as_bmp(binarized_labels, labels_path + "/")

train_input_data_dir = f"../input_data/train/"
train_output_dir = f"../sliced_images/train"
test_input_data_dir = f"../input_data/test/"
test_output_dir = f"../sliced_images/test"
mri_sequence_flags = ["Flair"]
labels_flags = ["3more", "OT"]

train_volume_paths = get_volume_paths(train_input_data_dir, mri_sequence_flags)
train_volume_paths = list(map(lambda x: x.replace("\\", "/"), train_volume_paths))

train_labels_paths = get_volume_paths(train_input_data_dir, labels_flags)
train_labels_paths = list(map(lambda x: x.replace("\\", "/"), train_labels_paths))

test_volume_paths = get_volume_paths(test_input_data_dir, mri_sequence_flags)
test_volume_paths = list(map(lambda x: x.replace("\\", "/"), test_volume_paths))

test_labels_paths = get_volume_paths(test_input_data_dir, labels_flags)
test_labels_paths = list(map(lambda x: x.replace("\\", "/"), test_labels_paths))

train_output_dir_names  = list(map(lambda x: x.split("/"), train_volume_paths))
train_output_dir_names  = list(map(lambda x: x[3], reversed(train_output_dir_names)))
test_output_dir_names  = list(map(lambda x: x.split("/"), test_volume_paths))
test_output_dir_names  = list(map(lambda x: x[3], reversed(test_output_dir_names)))

print("Creating training data.")
transform_volumes_into_slice_images(train_output_dir_names, train_volume_paths, train_labels_paths, train_output_dir, mri_sequence_flags[0], "Labels")
print("-"*25)
print("Creating testing data.")
transform_volumes_into_slice_images(test_output_dir_names, test_volume_paths, test_labels_paths, test_output_dir, mri_sequence_flags[0], "Labels")
print("-"*25)
