#define _CRT_SECURE_NO_WARNINGS

#include<opencv2/opencv.hpp>
#include<opencv2/ximgproc.hpp>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/ml.hpp>

#include<iostream>
#include<conio.h>
#include<Windows.h>
#include<tuple>
#include<thread>

#include"FeatureExtractor.h"
#include"CsvProcessing.h"
#include"SvmClassifier.h"

using namespace cv;
using namespace cv::ml;
using namespace cv::ximgproc;
using namespace std;

string fileExtension = "bmp";
double penalizationRatio = 0.50;
string mriSequenceFlag = "Flair";
string labelsFlag = "Labels";
string trainDataSourceDir = "..\\..\\sliced_images\\train";
string testDataSourceDir = "..\\..\\sliced_images\\test";
string trainDataCsvPath = "..\\..\\train_data.csv";
string testDataCsvPath = "..\\..\\test_data.csv";

string svmModelName = "svm_model";
int numberOfSamples = 0;
int histSize = 15;
int superpixelSize = 10;

void extractFeaturesScenario(string csvPath, string dataSourceDir, int numberOfImages) {
	string volumeCount = "";
	
	(numberOfImages == 0) ? (volumeCount = "all") : (volumeCount = std::to_string(numberOfImages));
	cout << "Extracting features from " << dataSourceDir << " using " << volumeCount << " volumes" << endl;

	FeatureExtractor extractor(fileExtension, penalizationRatio, mriSequenceFlag, labelsFlag, histSize, 10);
	extractor.extractFeaturesFromDataIntoCsv(csvPath, dataSourceDir, numberOfImages);
}

int main()
{  
	int scenarioNum = 0;
	std::ostringstream stringStream;
	stringStream << "This program processes image brain data using Opencv. To run a specific scenario, write coresponding number and press enter. " << endl
		<< "1:\tFeature extraction from training images using superpixels to csv" << endl
		<< "2:\tFeature extraction from testing images using superpixels to csv" << endl
		<< "3:\tFeature extraction and svm training" << endl
		<< "4:\tSVM prediction from csv" << endl
		<< "5:\tSVM prediction with features extraction from test data" << endl
		<< "6:\tFeature extraction, SVM training and prediction" << endl

		<< "Select the number of scenario: " << endl;
	std::string helpMessage = stringStream.str();
	cout << helpMessage;

	cin >> scenarioNum;

	switch (scenarioNum) {
	case 1: {
		extractFeaturesScenario(trainDataCsvPath, trainDataSourceDir, numberOfSamples);
		break;
	}
	case 2: {
		extractFeaturesScenario(testDataCsvPath, testDataSourceDir, numberOfSamples);
		break;
	}
		
	case 3: {
		FeatureExtractor extractor(fileExtension, penalizationRatio, mriSequenceFlag, labelsFlag, histSize, 5);
		SvmClassifier svm(svmModelName);
		CsvProcessing csvProc;
		svm.trainSvm(csvProc.readCsv(testDataCsvPath));
	}
			
	case 4: {
		SvmClassifier svm(svmModelName);
		svm.predictLabelsOfSuperpixelsFromCsv(trainDataCsvPath, 100000);
		break;
	}
			
	case 5: {
		FeatureExtractor extractor(fileExtension, penalizationRatio, mriSequenceFlag, labelsFlag, histSize, 10);
		SvmClassifier svm(svmModelName);
		svm.predictLabelsOfSuperpixels(testDataSourceDir, extractor, "Predictions", 5);
		break;
	}
			
	case 6: {
		FeatureExtractor extractor(fileExtension, penalizationRatio, mriSequenceFlag, labelsFlag, histSize, 5);
		extractor.extractFeaturesFromDataIntoCsv(trainDataCsvPath, trainDataSourceDir);
		SvmClassifier svm(svmModelName);
		CsvProcessing csvProc;
		svm.trainSvm(csvProc.readCsv(trainDataCsvPath));
		svm.predictLabelsOfSuperpixels(testDataSourceDir, extractor, "Predictions", 20);
		break;
	}
	}
	
	return 0;
}