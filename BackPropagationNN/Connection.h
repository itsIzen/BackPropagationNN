#pragma once
#include <vector>

class Connection {
public:
	Connection(double weight);
	~Connection();

	double m_weight;
	double m_delta;
private:
};