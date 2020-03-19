#ifndef UNIT_TEST_TEST_CELL
#define UNIT_TEST_TEST_CELL

#include "../NeuralNetwork/neuralNetwork.h"

class testNeuralNetwork : public neuralNetwork
{
public:
	class testCell : public cell
	{
	public:
		testCell(bool, int);
		void backwardPropagate(std::list < std::vector<float>>&, int, std::list<std::vector<float>>&, std::mutex&);
		void forwardPropagate(std::list < std::vector<float>>&, int);
	};

	class testNeuron : public neuron
	{
	public:
		testNeuron(bool, int);
	};
};

#endif
