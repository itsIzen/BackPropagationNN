#include "Layer.h"

Layer::Layer(int neuronsCount)
{
	m_neuronsCount = neuronsCount;
	for (int i = 0; i < neuronsCount; i++)
	{
		m_neurons.push_back(new Neuron());
	}
}

int Layer::GetNeuronsCount()
{
	return m_neuronsCount;
}

Neuron* Layer::GetNeuron(int index)
{
	/*if (index < m_neurons.size())
	{
		return m_neurons[index];
	}
	else
	{
		return nullptr;
	}*/
	return m_neurons[index];
}

Layer::~Layer()
{
}