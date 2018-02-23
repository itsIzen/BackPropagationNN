#pragma once
#include "Neuron.h"

class Layer {
public:
	Layer(int neuronsCount);
	~Layer();

	int GetNeuronsCount();

	Neuron* GetNeuron(int index);

private:
	std::vector<Neuron*> m_neurons;

	int m_neuronsCount;
};