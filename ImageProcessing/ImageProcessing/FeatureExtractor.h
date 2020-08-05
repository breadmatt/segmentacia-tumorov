#ifndef IMAGEPROCESSING_FEATUREEXTRACTOR_H
#define IMAGEPROCESSING_FEATUREEXTRACTOR_H

#pragma once
#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include <numeric>  
#include<tuple>
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/ml.hpp>
#include<opencv2/ximgproc.hpp>
#include<ctime>
#include"CsvProcessing.h"


using namespace std;
using namespace cv;
using namespace cv::ximgproc;

/* 
	This class is used feature extraction from superpixels.
	The defined features are:
	max, min, median, mean and standard deviation pixel intensity, x and y coords of superpixels centroid, z coord indicating slice number,
	or depth number of volume, histogram of superpixel
*/
class FeatureExtractor {
private:
	string extension;
	double penalisationRatio = 0.5;
	int histSize;
	bool useNormalization;
	/*
		Computes x and z coordinates of the centroid of the superpixel.
	*/
	cv::Vec2f computeCendtroidForSuperpixel(cv::Mat superpixelMask);

	/*
		Normalizes vector of feature using z-score.
	*/
	std::vector<float> normalizeVector(std::vector<float> inputVector);

	/*
		Computes median and standard deviation from the superpixel
	*/
	std::tuple<double, double> medianAndSdOfSuperpixel(cv::Mat superpixel, cv::Mat mask, double mean);

public:
	int superpixelSize = 10;

	string mriSequenceFlag = "Flair";
	string labelsFlag = "Labels";
	FeatureExtractor(string extension, double penalizationRatio, string mriSequenceFlag, string labelsFlag, int histSize = 15, int superpixelSize = 10, bool useNormalization = false);
	~FeatureExtractor();
	/*
		Gets the list of subdirectories from the parent directory.
	*/
	std::vector<std::string> getSubDirs(string sourceDir);

	/*
		Gets the list of files from the parent directory.
	*/
	std::vector<std::string> getFilesFromDir(string dir, string mriSequenceFlag);
	
	/*
		Crafts the features from the give images using superpixels, return the features and labels.
	*/
	tuple<vector<float>, int> extractFeaturesFromSuperpixel(cv::Mat inputImage, cv::Mat superpixelLabels, cv::Mat labelsMask, int superpixelId, int zPosition);
	
	/*
		Crafts the features from files from the give source directory and saves them into csv file, for the later use.
	*/
	void extractFeaturesFromDataIntoCsv(string savePath, string sourceDir, int volumeCount = 0);
	
	/*
		This function provides superpixel computation and return superpixel mask and labels.
	*/
	std::tuple<cv::Mat, cv::Mat, cv::Mat, int> applySuperpixelsOnImage(cv::Mat image);

	/*
		Decides the label of a given superpixels using ground truth mask. 
		The superpixel has label 1, if at least [ratio %] of pixels in the superpixels has 1 label in the ground truth mask.
	*/
	int getLabelFromGroundTruthMask(cv::Mat labelsSlice, cv::Mat mask, double ratio);
};

#endif //IMAGEPROCESSING_FEATUREEXTRACTOR_H