//Copyright(C) 2020 "Daniel Bramblett" <daniel.r.bramblett@gmail.com>
#ifndef NEURAL_NET_ERRORS
#define NEURAL_NET_ERRORS

#include<exception>

namespace NeuralNetwork
{
	//Thrown if two lists that should be the same length aren't.
	struct lists_not_same_length : public std::exception
	{

	};

	/*Thrown by the buildActFuncBundle() function when the provided string doesn't match any
	 *of the predefined activation functions.*/
	struct activation_function_not_found : public std::exception
	{

	};
}
#endif
