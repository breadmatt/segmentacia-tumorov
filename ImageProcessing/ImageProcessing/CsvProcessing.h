#ifndef IMAGEPROCESSING_CSVPROCESSING_H
#define IMAGEPROCESSING_CSVPROCESSING_H

#pragma once
#include<iostream>
#include<string>
#include<vector>
#include<fstream>
#include<sstream>
#include<tuple>


using namespace std;

/*
	This class is used for the manipulation with the csv files. 
	Has functions to read data from csv and write data into csv.
*/
class CsvProcessing
{
private:
public:
	CsvProcessing();
	~CsvProcessing();
	/*
		Function to write extracted features and labels into csv file
	*/
	void writeCsv(string filePath, vector<vector<float>> features, vector<int> labels, vector<string> columns, char delimeter = ',');
	
	
	/*
		Function to read features and labels from csv file into vector structures
	*/
	tuple<vector<vector<float>>, vector<int>> readCsv(string filePath,  char delimeter = ',');
};


#endif //IMAGEPROCESSING_CSVPROCESSING_H