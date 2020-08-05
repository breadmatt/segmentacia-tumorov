#include "SvmClassifier.h"



SvmClassifier::SvmClassifier(string svmModelName)
{
	this->svmModelName = svmModelName;

	this->svm = SVM::create();
	this->svm->setType(SVM::C_SVC);
	this->svm->setKernel(SVM::POLY);
	this->svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 4e7, 1e-6));
	this->svm->setGamma(3);
	this->svm->setDegree(3);
}


SvmClassifier::~SvmClassifier()
{
}

int SvmClassifier::calculateClassWeight(vector<int> labelsVect, int classVal) {
	int labelCount = std::count(labelsVect.begin(), labelsVect.end(), classVal);

	return labelsVect.size() / (2 * labelCount);
}

void SvmClassifier::trainSvm(tuple<vector<vector<float>>, vector<int>> featureTuple)
{
	// I learned how to use OpenCV SVM at https://docs.opencv.org/3.4/d1/d73/tutorial_introduction_to_svm.html
	// but the code is mine
	vector<vector<float>> featureVect; 
	vector<int> labelsVect;
	std::tie(featureVect, labelsVect) = featureTuple;

	Mat weights = (Mat_<float>(2, 1) << calculateClassWeight(labelsVect, 0), calculateClassWeight(labelsVect, 1));
	this->svm->setClassWeights(weights);

	float* featuresPtr = twoDimVectToOneDimPointer(featureVect);
	int* labelsPtr = vectToPointer(labelsVect);

	int numberOfSamples = featureVect.size();
	int numberOfFeatures = featureVect[0].size();

	std::cout << numberOfSamples << endl;

	Mat trainingDataMat(numberOfSamples, numberOfFeatures, CV_32F, featuresPtr);
	Mat labelsMat(numberOfSamples, 1, CV_32SC1, labelsPtr);

	featureVect.clear();
	labelsVect.clear();

	std::cout << "Training process starts." << endl;
	clock_t begin = clock();
	this->svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
	std::cout << "Training process finished." << endl;
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "elapsed time: " << elapsed_secs << " s" << endl;
	this->svm->save(this->svmModelName);

	delete[] featuresPtr;
	delete[] labelsPtr;
}

bool CreateFolder(string path)
{
	// I learned how to create folder in c++ here https://stackoverflow.com/questions/9235679/create-a-directory-if-it-doesnt-exist
	if (!CreateDirectory(path.c_str(), NULL))
	{
		std::cout << "Directory already exists." << endl;
		return false;
	}
	return true;
}

void SvmClassifier::predictLabelsOfSuperpixelsFromCsv(string csvPath, int numberOfSamples) {
	this->svm = cv::ml::StatModel::load<cv::ml::SVM>(this->svmModelName);
	vector<int> predictionsVect;

	CsvProcessing csvProc;
	vector<vector<float>> featureVect;
	vector<int> labelsVect;
	std::tie(featureVect, labelsVect) = csvProc.readCsv(csvPath);

	std::cout << featureVect[0].size() << endl;

	for (int i = 0; i < featureVect.size(); i++) {

		if (i == numberOfSamples){
			break;
		}

		Mat sampleMat = Mat(1, featureVect[i].size(), CV_32F);
		memcpy(sampleMat.data, featureVect[i].data(), featureVect[i].size() * sizeof(float));

		int prediction = this->svm->predict(sampleMat);
		predictionsVect.push_back(prediction);
		
		if (i % 1000 == 0) {
			std::cout << i << endl;
		}
	}
	Metrics metrics(predictionsVect, labelsVect);
	std::cout << "Accuracy: "<< metrics.computeAccuracy() <<endl;
	std::cout << "Precision: " << metrics.computePrecision() << endl;
	std::cout << "Recall: " << metrics.computeRecall() << endl;
}

int SvmClassifier::getMaxAreaContourId(vector<vector<cv::Point>> contours) {
	double maxArea = 0;
	int maxAreaContourId = -1;
	for (int j = 0; j < contours.size(); j++) {
		double newArea = cv::contourArea(contours.at(j));
		if (newArea > maxArea) {
			maxArea = newArea;
			maxAreaContourId = j;
		}
	}
	return maxAreaContourId;
}

void SvmClassifier::predictLabelsOfSuperpixels(string testDataSourceDir, FeatureExtractor extractor, string predictionSaveDir, int numberOfVolumes) {
	this->svm = cv::ml::StatModel::load<cv::ml::SVM>(this->svmModelName);
	std::cout << this->svm->getClassWeights() << endl;

	// if user wants only the subset of data
	vector<string> patientDirectories = extractor.getSubDirs(testDataSourceDir);
	if (numberOfVolumes > 0 && numberOfVolumes < patientDirectories.size()) {
		vector<string> sub(&patientDirectories[0], &patientDirectories[numberOfVolumes]);
		patientDirectories = sub;
	}
	// go through all patient dirs
	for (int i = 0; i < patientDirectories.size(); i++) {
		if (i < 7)
			continue;
		std::cout << patientDirectories[i] << endl;
		vector<string> mriSequenceFiles = extractor.getFilesFromDir(testDataSourceDir + "\\" + patientDirectories[i], extractor.mriSequenceFlag);
		vector<string> labelsFiles = extractor.getFilesFromDir(testDataSourceDir + "\\" + patientDirectories[i], extractor.labelsFlag);
		CreateFolder(testDataSourceDir + "\\" + patientDirectories[i] + "\\" + predictionSaveDir);

		clock_t begin = clock();

		std::vector<int> predictionVect;
		std::vector<int> groundTruthVect;
		
		// go through the images of the current patient
		for (int j = 0; j < mriSequenceFiles.size(); j++) {
			cv::Mat inputImage = imread(mriSequenceFiles[j], cv::IMREAD_GRAYSCALE);
			cv::Mat labelsImage = imread(labelsFiles[j], cv::IMREAD_GRAYSCALE);
			cv::Mat predictedImage(inputImage.cols, inputImage.rows, CV_8UC1);
			cv::Mat groundTruthMask(inputImage.cols, inputImage.rows, CV_8UC1);;
			cv::Mat superpixelsResult = inputImage.clone(), slicLabels, spMask, gtSpLabels;
			int numberOfSuperpixels;
			std::cout << "slice " << j << " / " << mriSequenceFiles.size() << endl;
			std::tie(superpixelsResult, slicLabels, spMask, numberOfSuperpixels) = extractor.applySuperpixelsOnImage(inputImage);
			gtSpLabels = Mat::zeros(predictedImage.size(), cv::IMREAD_GRAYSCALE);

			// go the superpixels of the current image
			for (int s = 0; s < numberOfSuperpixels; s++)
			{
				vector<float> features;
				int label;
				std::tie(features, label) = extractor.extractFeaturesFromSuperpixel(superpixelsResult, slicLabels, labelsImage, s, j);

				// features vector has not size 0, if it wasnt the superpixels of the background
				if (features.size() != 0) {
					Mat sampleMat = Mat(1, features.size(), CV_32F);
					memcpy(sampleMat.data, features.data(), features.size() * sizeof(float));
					
					//make the prediction
					int prediction = this->svm->predict(sampleMat);
					predictionVect.push_back(prediction);

					
					if (prediction == 1) {
						predictedImage.setTo(Scalar(255), (slicLabels == s));
					}
					else {
						predictedImage.setTo(Scalar(0), (slicLabels == s));
					}
					int groundTruthLabel = extractor.getLabelFromGroundTruthMask(labelsImage, slicLabels == s, 0.5);
					groundTruthVect.push_back(groundTruthLabel);
					if (groundTruthLabel) {
						groundTruthMask.setTo(Scalar(255), (slicLabels == s));
					}
				}
				else {
					predictedImage.setTo(Scalar(0), (slicLabels == s));
				}
			}

			// postprocessing 
			Mat element = getStructuringElement(cv::MORPH_ELLIPSE, Size(2 * extractor.superpixelSize + 1, 2 * extractor.superpixelSize + 1));
			cv::dilate(predictedImage, predictedImage, element);

			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(predictedImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

			int maxContourId = getMaxAreaContourId(contours);
			Mat result = Mat::zeros(predictedImage.size(), cv::IMREAD_GRAYSCALE);
			if (maxContourId >= 0) {
				cv::drawContours(result, contours, maxContourId, cv::Scalar(255), cv::FILLED);
			}

			imwrite(testDataSourceDir + "\\" + patientDirectories[i] + "\\" + predictionSaveDir + "\\" + to_string(j) + ".bmp", result);
		}

		Metrics metrics(predictionVect, groundTruthVect);
		std::cout << "Accuracy: " << metrics.computeAccuracy() << endl;
		std::cout << "Precision: " << metrics.computePrecision() << endl;
		std::cout << "Recall: " << metrics.computeRecall() << endl;

		clock_t end = clock();
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		std::cout << "elapsed time: " << elapsed_secs << " s" << endl;
	}
}

float** SvmClassifier::twoDimVectToTwoDimPointer(vector<vector<float>>& vect)
{
	float** retval = new float*[vect.size()];

	for (size_t i = 0; i < vect.size(); ++i)
	{
		retval[i] = new float[vect[i].size()];
		memcpy(retval[i], &vect[i][0], vect[i].size() * sizeof(float));
	}

	return retval;
}

int* SvmClassifier::vectToPointer(vector<int>& vect)
{
	int* retval = new int[vect.size()];

	memcpy(retval, &vect[0], vect.size() * sizeof(int));

	return retval;
}

float* SvmClassifier::twoDimVectToOneDimPointer(std::vector<std::vector<float>>& featureVect)
{
	size_t size = 0;

	for (const auto& vector : featureVect)
	{
		size += vector.size();
	}

	float* retval = new float[size];
	float* current = retval;

	for (size_t i = 0; i < featureVect.size(); ++i)
	{
		for (size_t j = 0; j < featureVect[i].size(); ++j)
		{
			*current++ = (float)featureVect[i][j];
		}
	}

	return retval;
}