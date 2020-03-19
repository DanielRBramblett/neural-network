//Copyright(C) 2020 "Daniel Bramblett" <daniel.r.bramblett@gmail.com>
/*Contains helper functions used in the NeuralNetwork project.*/
#ifndef HELPER_FUNCTIONS_NEURAL_NETWORK
#define HELPER_FUNCTIONS_NEURAL_NETWORK

#include<vector>

namespace NeuralNetwork
{
	/*Takes two same-length lists and adds the reference vector to the target vector while multiplying
	 *the value of the reference by a multiplier.*/
	void addVectors(std::vector<float> &target, const std::vector<float> &ref, const float multiplier);
}

#endif
