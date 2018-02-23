#include <iostream>

#include "NetTrainer.h"
#include <string>

void main() {
	std::cout << "hello" << std::endl;

	std::vector<double> inputs = { 0.0, 0.0 };
	std::vector<double> targetOutputs = { 1.0 };

	Net* net = Net::CreateNet(2, 6, 1);
	net->Output(inputs);
	for (int i = 0; i < net->m_outputs.size(); i++)
	{
		std::cout << "Untrained net output: " << net->m_outputs[i] << std::endl;
	}
	NetTrainer* trainer = new NetTrainer(net);
	trainer->TrainNet(inputs, targetOutputs);
	for (int i = 0; i < net->m_outputs.size(); i++)
	{
		std::cout << "Trained net output: " << net->m_outputs[i] << std::endl;
	}

	std::cin.get();
	return;
}