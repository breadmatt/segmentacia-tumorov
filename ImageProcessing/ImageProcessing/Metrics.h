#ifndef METRICS_H
#define METRICS_H

#pragma once
#include<vector>
#include<iostream>

/*
	This class is used for the confusion matrix computation, from which other metrices can be produced.
	The formulas was learned from https://en.wikipedia.org/wiki/Confusion_matrix.
*/

class Metrics
{
private:
	std::vector<int> predictedVector;
	std::vector<int> groundTruthVector;
	int **confusionMatrix;
	
public:
	Metrics(std::vector<int> predictedVector, std::vector<int> groundTruthVector);
	~Metrics();

	float computeAccuracy();
	float computeRecall();
	float computePrecision();
};

#endif //METRICS_H