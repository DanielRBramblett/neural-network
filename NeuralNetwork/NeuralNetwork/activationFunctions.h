//Copyright(C) 2020 "Daniel Bramblett" <daniel.r.bramblett@gmail.com>
/*Contains the prototypes for all the predefined activation functions along with the struct 
 *activationFunctionInfo that contains all the information needed on a certain activation function. 
 *This file also contains the prototype for the buildActFuncBundle which can create an instance
 *of the struct with all the filled-in information for any of the predefined activation functions.*/

#ifndef NN_PROVIDED_ACT_FUNC
#define NN_PROVIDED_ACT_FUNC

#include<functional>

namespace NeuralNetwork
{
	//The bundle that contains all the information needed on an activation function.
	struct activationFunctionInfo
	{
		//The activation function.
		std::function<float(const float)> activationFunction;
		//The gradient of the activation function.
		std::function<float(const float)> activationFunctionGradient;
		//Whether the gradient function is based on the activation function.
		/*For example, the gradient function, f'(x), of the sigmoid function can be written as:
		 *f'(x) = f(x)(1-f(x)) were f(x) is the sigmoid function.*/
		bool gradientInTermsOfFunc;
	};

	/*Given a string, attempts to find a predefined activation function that matches it and return an
	 *instance of the activationFunctionInfo struct containing all the information on that activation
	 *function. If one isn't found, the exception activation_function_not_found is thrown.*/
	activationFunctionInfo buildActFuncBundle(const std::string);

	//Sigmoid function and gradient function prototype.
	float sigmoid(const float);
	float sigmoidGrad(const float);
}

#endif
