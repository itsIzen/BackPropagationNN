#include "Net.h"
#include "ActivationFunctions.h"

Net::Net() : m_inputlayer(nullptr), m_hiddenlayer(nullptr), m_outputlayer(nullptr)
{
}

Net::~Net()
{
}

Net* Net::CreateNet(int inputsCount, int hiddenCount, int outputsCount)
{
	Net* net = new Net();

	net->m_inputlayer = new Layer(inputsCount);
	net->m_hiddenlayer = new Layer(hiddenCount);
	net->m_outputlayer = new Layer(outputsCount);

	net->m_outputs.resize(outputsCount);

	std::vector<Connection*> ihConnections;
	for (int j = 0; j < inputsCount; j++)
	{
		for (int k = 0; k < hiddenCount; k++)
		{
			ihConnections.push_back(new Connection(RandomizeWeight(-1.0f, 1.0f, 1000)));
		}
	}
	net->m_connections.push_back(ihConnections);

	std::vector<Connection*> hoConnections;
	for (int j = 0; j < hiddenCount; j++)
	{
		for (int k = 0; k < outputsCount; k++)
		{
			hoConnections.push_back(new Connection(RandomizeWeight(-1.0f, 1.0f, 1000)));
		}
	}
	net->m_connections.push_back(hoConnections);

	return net;
}

void Net::CreateNetFromData(std::string fataPath)
{
}

void Net::CalculateHiddenLayer()
{
	for (int hiddenIndex = 0; hiddenIndex < m_hiddenlayer->GetNeuronsCount(); hiddenIndex++)
	{
		double sum = 0.0;
		for (int inputIndex = 0; inputIndex < m_inputlayer->GetNeuronsCount(); inputIndex++)
		{
			sum += m_inputlayer->GetNeuron(inputIndex)->m_value * m_connections[0][hiddenIndex*m_inputlayer->GetNeuronsCount() + inputIndex]->m_weight;
		}
		m_hiddenlayer->GetNeuron(hiddenIndex)->m_value = ActivationFunctions::sigmoid(sum);
	}
}

void Net::CalculateOutputLayer()
{
	for (int outputIndex = 0; outputIndex < m_outputlayer->GetNeuronsCount(); outputIndex++)
	{
		double sum = 0.0;
		for (int hiddenIndex = 0; hiddenIndex < m_hiddenlayer->GetNeuronsCount(); hiddenIndex++)
		{
			sum += m_hiddenlayer->GetNeuron(hiddenIndex)->m_value * m_connections[1][outputIndex*m_outputlayer->GetNeuronsCount() + hiddenIndex]->m_weight;
		}
		m_outputlayer->GetNeuron(outputIndex)->m_value = ActivationFunctions::sigmoid(sum);
	}
}

void Net::Output(std::vector<double> inputs)
{
	for (int i = 0; i < m_inputlayer->GetNeuronsCount(); i++)
	{
		m_inputlayer->GetNeuron(i)->m_value = inputs[i];
	}

	CalculateHiddenLayer();
	CalculateOutputLayer();

	m_outputs.clear();
	for (int i = 0; i < m_outputlayer->GetNeuronsCount(); i++)
	{
		m_outputs.push_back(m_outputlayer->GetNeuron(i)->m_value);
	}
}