#include "NetTrainer.h"

NetTrainer::NetTrainer(Net* targetNet)
{
	m_activeNet = targetNet;

	m_learningRate = 0.1;
	m_momentum = 0.9;
	m_targetError = 1E-9;

	m_currentEpoch = 0;
	m_maxEpoch = 50000;

	m_useBatchLearning = false;
}

NetTrainer::~NetTrainer()
{
}

void NetTrainer::SetLearningRate(double value)
{
	m_learningRate = value;
}

void NetTrainer::SetMomentum(double value)
{
	m_momentum = value;
}

void NetTrainer::SetTargetError(double value)
{
	m_targetError = value;
}

void NetTrainer::TrainNet(std::vector<double> inputs, std::vector<double> targetOutputs)
{
	m_errorGradientsHidden.clear();
	m_errorGradientsOutput.clear();
	for (int i = 0; i < m_activeNet->m_hiddenlayer->GetNeuronsCount(); i++) {
		m_errorGradientsHidden.push_back(1);
	}
	for (int i = 0; i < targetOutputs.size(); i++) {
		m_errorGradientsOutput.push_back(1);
	}

	double trainError = 1.0;
	m_currentEpoch = 0;

	do
	{
		m_activeNet->Output(inputs);
		Backpropagate(targetOutputs);

		m_currentEpoch++;
	} while (m_currentEpoch < m_maxEpoch || trainError < m_targetError);
}

void NetTrainer::UpdateWeights()
{
	for (auto inputIndex = 0; inputIndex < m_activeNet->m_inputlayer->GetNeuronsCount(); inputIndex++)
	{
		for (auto hiddenIndex = 0; hiddenIndex < m_activeNet->m_hiddenlayer->GetNeuronsCount(); hiddenIndex++)
		{
			m_activeNet->m_connections[0][inputIndex*m_activeNet->m_hiddenlayer->GetNeuronsCount() + hiddenIndex]->m_weight +=
				m_activeNet->m_connections[0][inputIndex*m_activeNet->m_hiddenlayer->GetNeuronsCount() + hiddenIndex]->m_delta;

			if (m_useBatchLearning)
			{
				m_activeNet->m_connections[0][inputIndex*m_activeNet->m_hiddenlayer->GetNeuronsCount() + hiddenIndex]->m_delta = 0;
			}
		}
	}

	for (auto hiddenIndex = 0; hiddenIndex < m_activeNet->m_hiddenlayer->GetNeuronsCount(); hiddenIndex++)
	{
		for (auto outputIndex = 0; outputIndex < m_activeNet->m_outputlayer->GetNeuronsCount(); outputIndex++)
		{
			m_activeNet->m_connections[1][hiddenIndex*m_activeNet->m_outputlayer->GetNeuronsCount() + outputIndex]->m_weight +=
				m_activeNet->m_connections[1][hiddenIndex*m_activeNet->m_outputlayer->GetNeuronsCount() + outputIndex]->m_delta;

			if (m_useBatchLearning)
			{
				m_activeNet->m_connections[1][hiddenIndex*m_activeNet->m_outputlayer->GetNeuronsCount() + outputIndex]->m_delta = 0;
			}
		}
	}
}

void NetTrainer::Backpropagate(std::vector<double> targetOutputs)
{
	// Update deltas between hidden and output layers
	for (int outputIndex = 0; outputIndex < m_activeNet->m_outputlayer->GetNeuronsCount(); outputIndex++)
	{
		// Get error gradient
		m_errorGradientsOutput[outputIndex] = CalculateOutputGradient(targetOutputs[outputIndex], m_activeNet->m_outputlayer->GetNeuron(outputIndex)->m_value);

		for (int hiddenIndex = 0; hiddenIndex < m_activeNet->m_hiddenlayer->GetNeuronsCount(); hiddenIndex++)
		{
			// Calculate weight delta
			if (m_useBatchLearning)
			{
				m_activeNet->m_connections[1][outputIndex*m_activeNet->m_hiddenlayer->GetNeuronsCount() + hiddenIndex]->m_delta +=
					m_learningRate * m_activeNet->m_hiddenlayer->GetNeuron(hiddenIndex)->m_value * m_errorGradientsOutput[outputIndex];
			}
			else
			{
				m_activeNet->m_connections[1][outputIndex*m_activeNet->m_hiddenlayer->GetNeuronsCount() + hiddenIndex]->m_delta =
					m_learningRate * m_activeNet->m_hiddenlayer->GetNeuron(hiddenIndex)->m_value  * m_errorGradientsOutput[outputIndex] +
					m_momentum * m_activeNet->m_connections[1][outputIndex*m_activeNet->m_hiddenlayer->GetNeuronsCount() + hiddenIndex]->m_delta;
			}
		}
	}

	// Update deltas between input and hidden layers
	for (int hiddenIndex = 0; hiddenIndex < m_activeNet->m_hiddenlayer->GetNeuronsCount(); hiddenIndex++)
	{
		// Get error gradient
		m_errorGradientsHidden[hiddenIndex] = CalculateHiddenGradient(hiddenIndex);

		for (int inputIndex = 0; inputIndex < m_activeNet->m_inputlayer->GetNeuronsCount(); inputIndex++)
		{
			// Calculate weight delta
			if (m_useBatchLearning)
			{
				m_activeNet->m_connections[0][hiddenIndex* m_activeNet->m_inputlayer->GetNeuronsCount() + inputIndex]->m_delta +=
					m_learningRate * m_activeNet->m_inputlayer->GetNeuron(inputIndex)->m_value * m_errorGradientsHidden[hiddenIndex];
			}
			else
			{
				m_activeNet->m_connections[0][hiddenIndex* m_activeNet->m_inputlayer->GetNeuronsCount() + inputIndex]->m_delta =
					m_learningRate * m_activeNet->m_inputlayer->GetNeuron(inputIndex)->m_value * m_errorGradientsHidden[hiddenIndex] +
					m_momentum * m_activeNet->m_connections[0][hiddenIndex* m_activeNet->m_inputlayer->GetNeuronsCount() + inputIndex]->m_delta;
			}
		}
	}

	if (!m_useBatchLearning)
	{
		UpdateWeights();
	}
}