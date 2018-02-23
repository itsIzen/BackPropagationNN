#pragma once
#include <cmath>
#include <random>

inline double RandomizeWeight(double min, double max, int seed)
{
	std::random_device rd;   // non-deterministic generator
	std::mt19937 gen(rd()*seed);
	std::uniform_real_distribution<> dist(min, max);

	//return min + (static_cast<double>(dist(gen)) * static_cast<double>(max - min)) / RAND_MAX;
	return static_cast<double>(dist(gen));
}

class ActivationFunctions {
public:

	static double sigmoid(double x) {
		return 1.0 / (1.0 + exp(-x));
	};

	static double sigmoidDerivative(double x) {
		return x * (1.0 - x);
	};
};