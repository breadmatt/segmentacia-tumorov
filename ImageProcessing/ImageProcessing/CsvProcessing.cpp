#include "CsvProcessing.h"



CsvProcessing::CsvProcessing()
{
}


CsvProcessing::~CsvProcessing()
{
}

void CsvProcessing::writeCsv(string filePath, vector<vector<float>> features, vector<int> labels, vector<string> columns, char delimeter)
{
	std::ofstream file;
	file.open(filePath);
	string serialized = "";
	for (int i = 0; i < columns.size(); i++) {
		serialized += columns[i];
		if(i < columns.size() - 1){
			serialized += delimeter;
		}
	}
	serialized += "\n";
	file << serialized;

	for (int i = 0; i < features.size(); i++) {
		serialized = "";
		for (int j = 0; j < features[i].size(); j++) {
			serialized += to_string(features[i][j]) + delimeter;
			
		}
		serialized += to_string(labels[i]) + "\n";
		file << serialized;
	}
	file.close();
}

tuple<vector<vector<float>>, vector<int>> CsvProcessing::readCsv(string filePath, char delimeter)
{
	vector<vector<float>> vectOfFeatures;
	vector<int> labelsVect;
	ifstream file(filePath);
	string line = "";

	int firstLine = 0;
	int numberOfCols = 0;
	while (std::getline(file, line))
	{
		std::stringstream   lineStream(line);
		string cell = "";
		if (firstLine == 0) {
			firstLine++;
			vector<string> columns;
			while (std::getline(lineStream, cell, delimeter))
			{
				numberOfCols++;
			}
		}
		else {
			int i = 0;
			vector<float> features;
			std::stringstream   lineStream(line);
			string cell = "";
			while (std::getline(lineStream, cell, delimeter))
			{
				if (i < numberOfCols - 1) {
					features.push_back(stof(cell));
					i++;
				}
				else {
					labelsVect.push_back(stoi(cell));
				}
			}
			vectOfFeatures.push_back(features);
		}
		
	}
	file.close();
	for (std::vector<float>::const_iterator i = vectOfFeatures[0].begin(); i != vectOfFeatures[0].end(); ++i)
		std::cout << *i << "\t";
	cout << endl;
	for (std::vector<float>::const_iterator i = vectOfFeatures[1].begin(); i != vectOfFeatures[1].end(); ++i)
		std::cout << *i << "\t";
	cout << endl;

	return { vectOfFeatures, labelsVect };
}