#include "Metrics.h"

Metrics::Metrics(std::vector<int> predictedVector, std::vector<int> groundTruthVector)
{
	this->confusionMatrix = new int*[2];

	for (size_t i = 0; i < 2; i++)
	{
		this->confusionMatrix[i] = new int[2];
	}
	int size = predictedVector.size();
	for (int i = 0; i < size; i++)
	{
		this->confusionMatrix[predictedVector[i]][groundTruthVector[i]]++;
	}
}

Metrics::~Metrics()
{
	delete[] this->confusionMatrix;
}

float Metrics::computeAccuracy() 
{
	float tptn = (this->confusionMatrix[0][0] + this->confusionMatrix[1][1]); //numerator
	float tptnfpfn = (this->confusionMatrix[0][0] + this->confusionMatrix[1][1] + this->confusionMatrix[1][0] + this->confusionMatrix[0][1]); //denominator
	return (float)tptn / tptnfpfn;
}
float Metrics::computeRecall() 
{
	return (float)this->confusionMatrix[0][0] / (this->confusionMatrix[0][0] + this->confusionMatrix[1][0]);
}
float Metrics::computePrecision() 
{
	return (float)this->confusionMatrix[0][0] / (this->confusionMatrix[0][0] + this->confusionMatrix[0][1]);
}