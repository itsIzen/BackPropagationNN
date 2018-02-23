#pragma once

#include "Net.h"

class NetTrainer
{
public:
	NetTrainer(Net* targetNet);
	~NetTrainer();

	void SetLearningRate(double value);
	void SetMomentum(double value);
	void SetTargetError(double value);

	void TrainNet(std::vector<double> inputs, std::vector<double> target_outputs);

	inline double CalculateOutputGradient(double targetValue, double outputValue) const
	{
		return outputValue * (1.0 - outputValue) * (targetValue - outputValue);
	}
	inline double CalculateHiddenGradient(int neuronIndex) const
	{
		// Get sum of hidden->output weights * output error gradients
		double sum = 0;
		for (auto outputIdx = 0; outputIdx < m_activeNet->m_outputlayer->GetNeuronsCount(); outputIdx++)
		{
			sum += m_activeNet->m_connections[1][neuronIndex*m_activeNet->m_outputlayer->GetNeuronsCount() + outputIdx]->m_weight * m_errorGradientsOutput[outputIdx];
		}

		// Return error gradient
		return m_activeNet->m_hiddenlayer->GetNeuron(neuronIndex)->m_value * (1.0 - m_activeNet->m_hiddenlayer->GetNeuron(neuronIndex)->m_value) * sum;
	}

	void Backpropagate(std::vector<double> targetOutputs);
	void UpdateWeights();

private:
	Net * m_activeNet;

	double m_learningRate; // DEFAULT: 0.1;
	double m_momentum; // DEFAULT: 0.9;
	double m_targetError; // Error value, used to finish network train. DEFAULT: 1E-6

	unsigned int m_maxEpoch; // Max epoch number for training. DEFAULT: 50000
	unsigned int m_currentEpoch;

	bool m_useBatchLearning;

	std::vector<double> m_errorGradientsHidden;     // Error gradients for the hidden layer
	std::vector<double> m_errorGradientsOutput;     // Error gradients for the outputs
};
