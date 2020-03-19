#include "helperFunctions.h"
#include "neuralNetworkErrors.h"
#include "preprocessorFlags.h"

void addVectors(std::vector<float> &target, const std::vector<float> &ref, const float weight)
{
	std::vector<float>::iterator targetIt = target.begin();
	std::vector<float>::const_iterator refIt = ref.begin();
#if SAFE_CELL
	for (; targetIt != target.end() && refIt != ref.end(); ++targetIt, ++refIt)
	{
		*targetIt += *refIt * weight;
	}
	if (targetIt != target.end() ^ refIt != ref.end())
	{
		throw lists_not_same_length();
	}
#else
	for (; targetIt != target.end(); ++targetIt, ++refIt)
	{
		*targetIt += *refIt * weight;
	}
#endif
}