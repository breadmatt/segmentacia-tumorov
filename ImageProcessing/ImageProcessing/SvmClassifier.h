#ifndef IMAGEPROCESSING_SVMCLASSIFIER_H
#define IMAGEPROCESSING_SVMCLASSIFIER_H

#pragma once

#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/ml.hpp>
#include<iostream>
#include<vector>
#include <algorithm>
#include "FeatureExtractor.h"
#include"CsvProcessing.h"
#include"Metrics.h"
#include<ctime>

using namespace std;
using namespace cv;
using namespace cv::ml;


/*
	This class is used for the SVM model training, used for the superpixel classification.
	Tranined model can be used for predicting labels for superpixels of new data.
*/
class SvmClassifier
{
private:
	string svmModelName;

	/*
		Conversion of vector of vectors into 2D pointer, used by the SVM.
	*/
	float** twoDimVectToTwoDimPointer(vector<vector<float>>& vect);

	/*
		Conversion of vector of vectors into 1D pointer, used by the SVM.
	*/
	float* twoDimVectToOneDimPointer(std::vector<std::vector<float>>& featureVect);

	/*
		Conversion of vector into 1D pointer, used by the SVM.
	*/
	int* vectToPointer(vector<int>& vect);
	
	
	/*
		Calculates the class weights for the given class in the labels vector. The class weight
		indicates the penalisation of the class by the SVM, if the dataset is imbalanced.
	*/
	int calculateClassWeight(vector<int> labelsVect, int classVal);
	
	
	/*
		This function finds the largest contour in the segmentatinon mask, measuring by its area
	*/
	int getMaxAreaContourId(vector<vector<cv::Point>> contours);
	Ptr<SVM> svm;

public:
	SvmClassifier(string svmModelName);
	~SvmClassifier();
	/*
		Function that trains SVM model from given features and labels
	*/
	void trainSvm(tuple<vector<vector<float>>, vector<int>> featureTuple);
	
	
	/*
		Function that predicts labels from a trained model from input image data. Function uses extractor to extract features 
		from input data superpixels. Extracted features are passed to SVM, which predicts labels for superpixels. Images with predicted labels
		for each superpixels are saved into directory ..//..//sliced_images//test//pacient_id//Predictions
	*/
	void predictLabelsOfSuperpixels(string testDataSourceDir, FeatureExtractor extractor, string predictionSaveDir, int numberOfVolumes = 0);
	
	
	
	/*
		Function that predicts labels from a trained model using csv file with extracted features and labels from superpixels.
		It is meant to be good for evaluation purposes.
	*/
	void predictLabelsOfSuperpixelsFromCsv(string csvPath, int numberOfSamples);
};

#endif //IMAGEPROCESSING_SVMCLASSIFIER_H