#include "activationFunctions.h"
#include "neuralNetworkErrors.h"
#include<math.h>

activationFunctionInfo buildActFuncBundle(const std::string activationFunctionName)
{
	activationFunctionInfo output;
	if (activationFunctionName == "sigmoid")
	{
		output.activationFunction = sigmoid;
		output.activationFunctionGradient = sigmoidGrad;
		output.gradientInTermsOfFunc = true;
	}
	else
	{
		throw activation_function_not_found();
	}
	return output;
}

float sigmoid(const float input)
{
	//TODO: add safety check for extremely high or low float values to prevent overflow.
	return 1 / (1 + exp(-input));
}

float sigmoidGrad(const float input)
{
	return input * (1 - input);
}