#!/bin/bash

export PATH="$PATH:/c/ProgramData/Anaconda3"

volumes_dir="input_data"
slices_dir="sliced_images"
if [ -d "$volumes_dir" ]; then rm -Rf $volumes_dir; fi
if [ -d "$slices_dir" ]; then rm -Rf $slices_dir; fi


mkdir -p $volumes_dir;
test_zip="$volumes_dir/test.zip"
test_dir="$volumes_dir/test"
train_zip="$volumes_dir/train.zip"
train_dir="$volumes_dir/train"

if [ ! -f $test_zip ] || [ ! -d $test_dir ]; 
then
    echo "$test_zip not found, download starting"
	python gdownloader.py 'https://drive.google.com/uc?id=1AcN5g_fBn94Xq9Cp9q3WKDkAG9WTiXUF' $test_zip
	unzip $test_zip -d $volumes_dir
	rm $test_zip
fi

if [ ! -f $train_zip ] || [ ! -d $train_dir ]; 
then
    echo "$train_zip not found, download starting"
	python gdownloader.py 'https://drive.google.com/uc?id=1UTCxbVSUhgbxWyMj0-gxR7EIncqB5y9p' $train_zip
	unzip $train_zip -d $volumes_dir
	rm $train_zip
fi

echo
echo "Transforming images into slices. This may take a while."
cd VolumeTools && python volumes_loader.py;

echo
echo "Finished. Press any key."
read