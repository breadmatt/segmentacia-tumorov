#include"FeatureExtractor.h"

FeatureExtractor::FeatureExtractor(string extension, double penalizationRatio, string mriSequenceFlag, string labelsFlag, int histSize, int superpixelSize, bool useNormalization) {
	this->extension = extension;
	this->penalisationRatio = penalisationRatio;
	this->mriSequenceFlag = mriSequenceFlag;
	this->labelsFlag = labelsFlag;
	this->histSize = histSize; // number of bins
	this->superpixelSize = superpixelSize;
	this->useNormalization = useNormalization;
}

FeatureExtractor::~FeatureExtractor() {}

vector<string> FeatureExtractor::getSubDirs(string sourceDir) {				// I used this code from external sorce at https://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c
	std::vector<std::string> subdirectories;
	string search_path = sourceDir + "\\*";
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && fd.cFileName[0] != '.') {
				subdirectories.push_back(fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return subdirectories;
}

std::vector<std::string> FeatureExtractor::getFilesFromDir(string dir, string mriSequenceFlag) {
	vector<string> names;													// from this line
	dir += "\\" + mriSequenceFlag + "\\";									// I found this piece of code at https://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c
	string search_path = dir + "\\*.*";
	WIN32_FIND_DATA fd;														
	HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				names.push_back(fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
																			// to this line
	vector<int> numbers;
	for (int i = 0; i < names.size(); i++) {
		std::string token = names[i].substr(0, names[i].find("."));
		numbers.push_back(std::stoi(token));
	}
	std::sort(numbers.begin(), numbers.end());
	vector<string> sorted;
	for (int i = 0; i < names.size(); i++) {
		sorted.push_back(dir + std::to_string(numbers[i]) + "." + extension);
	}
	return sorted;
}

Vec2f FeatureExtractor::computeCendtroidForSuperpixel(cv::Mat superpixelMask) {
	cv::Vec2f centroid;

	Moments m = moments(superpixelMask, true);
	Point p1(m.m10 / m.m00, m.m01 / m.m00);
	centroid.val[0] = (double)p1.y / superpixelMask.cols;
	centroid.val[1] = (double)p1.x / superpixelMask.rows;

	return centroid;
}

int FeatureExtractor::getLabelFromGroundTruthMask(Mat labelsSlice, Mat mask, double ratio) {
	Mat gtMask = mask.clone();
	Mat superpixelLabels;
	labelsSlice.copyTo(superpixelLabels, gtMask);
	int numberOfPixelsInSuperpixelMask = cv::countNonZero(gtMask);
	int numberOfPixelsInSuperpixelLabels = cv::countNonZero(superpixelLabels);
	// if the superpixel mask overlaps with the ground truth corresponding region, the superpixels has label 1 denoting the tumor
	if (numberOfPixelsInSuperpixelLabels >= numberOfPixelsInSuperpixelMask * ratio) {
		return 1;
	}
	else {
		//otherwise it is not tumor
		return 0;
	}
}

std::tuple<double, double> FeatureExtractor::medianAndSdOfSuperpixel(Mat superpixel, Mat mask, double mean) {
	std::vector<double> pixels;
	double variance = 0;
	int pixNumber = 0;
	for (int i = 0; i < superpixel.rows; i++) {
		for (int j = 0; j < superpixel.cols; j++) {
			if (mask.at<uchar>(i, j) > 0) {
				int pixVal = superpixel.at<uchar>(i, j);
				pixels.push_back(pixVal);
				variance += std::pow(pixVal - mean, 2);
				pixNumber++;
			}
		}
	}
	variance /= pixNumber;

	std::sort(pixels.begin(), pixels.end());
	int size = pixels.size(), median;
	if (size == 0) {
		median = 0;
	}
	else {
		if (size % 2 == 0) {
			median = (pixels[size / 2 - 1] + pixels[size / 2]) / 2;
		}
		else {
			median = pixels[size / 2];
		}
	}
	return  { median, std::sqrt(variance) };
}

vector<float> FeatureExtractor::normalizeVector(vector<float> inputVector) {

	float mean = std::accumulate(std::begin(inputVector), std::end(inputVector), 0.0) / inputVector.size();
	float sq_sum = std::inner_product(std::begin(inputVector), std::end(inputVector), std::begin(inputVector), 0.0);
	float stdev = std::sqrt(sq_sum / inputVector.size() - mean * mean);
	//z-score normalisation
	for (int i = 0; i < inputVector.size(); i++) {
		inputVector[i] = (inputVector[i] - mean) / stdev;
	}

	return inputVector;
}

tuple<vector<float>, int> FeatureExtractor::extractFeaturesFromSuperpixel(Mat inputImage, Mat superpixelLabels, Mat labelsMask, int superpixelId, int zPosition) {
	Mat single_superpixels_mask = (superpixelLabels == superpixelId), result, slice = inputImage.clone();
	double min, max, median, sd;
	slice.copyTo(result, single_superpixels_mask);

	cv::minMaxLoc(result, &min, &max, NULL, NULL, single_superpixels_mask);
	
	Scalar mean_intensity = cv::mean(result, single_superpixels_mask);
	
	int numberOfpixelsWithinSuperpixel = cv::countNonZero(single_superpixels_mask);
	
	std::tie(median, sd) = medianAndSdOfSuperpixel(result, single_superpixels_mask, mean_intensity.val[0]);
	
	int label = getLabelFromGroundTruthMask(labelsMask, single_superpixels_mask, penalisationRatio);
	
	auto centroid = computeCendtroidForSuperpixel(single_superpixels_mask);

	cv::Mat histogram;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true, accumulate = false;
	cv::calcHist(&inputImage, 1, 0, single_superpixels_mask, histogram, 1, &histSize, &histRange, uniform, accumulate);

	vector<float> features;

	//if the max value is 0, it means the superpixels contains only background
	if (max > 0) {
		features.push_back(min);
		features.push_back(max);
		features.push_back(mean_intensity.val[0]);
		features.push_back(median);
		features.push_back(sd);
		features.push_back(numberOfpixelsWithinSuperpixel);
		features.push_back(centroid.val[0]);
		features.push_back(centroid.val[1]);
		features.push_back(zPosition);
		for (int x = 0; x < histSize; x++) {
			features.push_back(histogram.at<float>(x));
		}
		if (this->useNormalization) {
			features = normalizeVector(features);
		}
	}
	return { features, label };
}

tuple<Mat, Mat, Mat, int> FeatureExtractor::applySuperpixelsOnImage(Mat image) {
	// I learned how to use superpixels in OpenCV here: https://docs.opencv.org/3.4/df/d6c/group__ximgproc__superpixel.html
	// but the code is mine
	Mat mask, result = image.clone(), slicLabels;
	cv::Ptr<SuperpixelSLIC> superpixelsPtr = cv::ximgproc::createSuperpixelSLIC(image.clone(), cv::ximgproc::SLIC, this->superpixelSize, 50.0f);
	superpixelsPtr->iterate(10);
	superpixelsPtr->getLabelContourMask(mask, true);
	result.setTo(Scalar(0, 0, 255), mask);
	superpixelsPtr->getLabels(slicLabels);
	int numberOfSuperpixels = superpixelsPtr->getNumberOfSuperpixels();

	return {result, slicLabels, mask, numberOfSuperpixels};
}


void FeatureExtractor::extractFeaturesFromDataIntoCsv(string savePath, string sourceDir, int volumeCount) {

	vector<string> subdirectories = getSubDirs(sourceDir);
	vector<int> outputLabels;
	vector<vector<float>> outputFeatures;

	// if user wants only the subset of data
	if (volumeCount > 0 && volumeCount < subdirectories.size()) {
		vector<string> sub1(&subdirectories[0], &subdirectories[volumeCount]);
		subdirectories = sub1;
	}
	int subDirSize = subdirectories.size();

	// go through all patient dirs
	for (int i = 0; i < subDirSize; i++) {
		vector<string> mriSequenceFiles = getFilesFromDir(sourceDir + "\\" + subdirectories[i], mriSequenceFlag);
		vector<string> labelsFiles = getFilesFromDir(sourceDir + "\\" + subdirectories[i], labelsFlag);

		
		clock_t begin = clock();
		std::cout << i + 1 << " / " << subDirSize << "\t";

		// go through the image files
		for (int j = 0; j < mriSequenceFiles.size(); j++) {
			cv::Mat inputImage = imread(mriSequenceFiles[j], cv::IMREAD_GRAYSCALE);
			cv::Mat labelsImage = imread(labelsFiles[j], cv::IMREAD_GRAYSCALE);
			cv::Mat result = inputImage.clone(), slicLabels, spMask;
			int numberOfSuperpixels;

			std::tie(result, slicLabels, spMask, numberOfSuperpixels) = applySuperpixelsOnImage(inputImage);

			// go through all superpixels
			for (int s = 0; s < numberOfSuperpixels; s++)
			{
				vector<float> features;
				int label;
				std::tie(features, label) = extractFeaturesFromSuperpixel(result, slicLabels, labelsImage, s, j);
				// features vector size is equal to 0, if its max intensity value was 0 and it was only the background 
				if (features.size() != 0) {
					outputFeatures.push_back(features);
					outputLabels.push_back(label);
				}
			}
		}
		clock_t end = clock();
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		std::cout << "elapsed time: " << elapsed_secs << " s" << endl;
	}
	vector<string> columns = { "min", "max", "mean", "median", "sd", "pixelsInSuperpixel", "xcenter", "ycenter", "z pos" };

	for (int x = 0; x < this->histSize; x++) {
		columns.push_back("hist_bin_" + to_string(x));
	}
	columns.push_back("label");
	CsvProcessing csvProc = CsvProcessing();
	csvProc.writeCsv(savePath, outputFeatures, outputLabels, columns);
}