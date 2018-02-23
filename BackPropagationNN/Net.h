#pragma once
#include "Layer.h"

class Net {
public:
	static Net* CreateNet(int inputsCount, int hiddenCount, int outputsCount);
	void CreateNetFromData(std::string fataPath);

	void CalculateHiddenLayer();
	void CalculateOutputLayer();

	void Output(std::vector<double> inputs);

	Layer* m_inputlayer;	//input layer of the network
	Layer* m_hiddenlayer;	// hidden layer
	Layer* m_outputlayer;	//output layer

	std::vector<double> m_outputs;
	std::vector<std::vector<Connection*>> m_connections;

private:
	Net();
	~Net();
};