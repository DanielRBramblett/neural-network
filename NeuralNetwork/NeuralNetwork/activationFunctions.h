#ifndef NN_PROVIDED_ACT_FUNC
#define NN_PROVIDED_ACT_FUNC

#include<functional>

struct activationFunctionInfo
{
	std::function<float(const float)> activationFunction;
	std::function<float(const float)> activationFunctionGradient;
	bool gradientInTermsOfFunc;
};

activationFunctionInfo buildActFuncBundle(const std::string);

float sigmoid(const float);
float sigmoidGrad(const float);

#endif
