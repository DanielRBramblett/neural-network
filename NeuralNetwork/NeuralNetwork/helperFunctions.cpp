//Copyright(C) 2020 "Daniel Bramblett" <daniel.r.bramblett@gmail.com>
#include "helperFunctions.h"
#include "neuralNetworkErrors.h"
#include "preprocessorFlags.h"

namespace NeuralNetwork
{
	void addVectors(std::vector<float> &target, const std::vector<float> &ref, const float multiplier)
	{
		std::vector<float>::iterator targetIt = target.begin();
		std::vector<float>::const_iterator refIt = ref.begin();
#if SAFE_CELL
		for (; targetIt != target.end() && refIt != ref.end(); ++targetIt, ++refIt)
		{
			*targetIt += *refIt * multiplier;
		}
		if (targetIt != target.end() ^ refIt != ref.end())
		{
			throw lists_not_same_length();
		}
#else
		for (; targetIt != target.end(); ++targetIt, ++refIt)
		{
			*targetIt += *refIt * multiplier;
		}
#endif
	}
}