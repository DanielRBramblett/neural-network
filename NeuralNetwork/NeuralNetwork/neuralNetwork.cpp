//Copyright(C) 2020 "Daniel Bramblett" <daniel.r.bramblett@gmail.com>
#include "neuralNetwork.h"
#include "neuralNetworkErrors.h"
#include "helperFunctions.h"
#include<numeric>
#include<iostream>

namespace NeuralNetwork
{
	//Nested classes implementations.
	//cell:
	neuralNetwork::cell::cell(bool propFurther, int newIndex) :backPropagateFurther(propFurther), cellIndex(newIndex)
	{
#if SAFE_CELL
		if (newIndex < 0)
		{
			cellIndex = 0;
			throw std::out_of_range("A negativate index isn't valid.");
		}
#endif
	}

	/*Attempts to add a connection between this cell and another cell. If the connection already
	  exists, the connections are not changed and this function returns false. Otherwise, this
	  function will return true. */
	bool neuralNetwork::cell::addConnection(int connectionIndex)
	{
		//Checks if the inputted index is a possible index 
#if SAFE_CELL
		if (connectionIndex < 0)
		{
			throw std::out_of_range("A negative index isn't valid.");
		}
#endif

		/*Iterates over the list of connections to see if this connection already exists. Otherwise,
		  when it finds a connection with a higher index, the new connection is added.*/
		std::list<int>::iterator it = connections.begin();
		for (; it != connections.end(); ++it)
		{
			if (*it == connectionIndex)
			{
				return false;
			}
			if (*it > connectionIndex)
			{
				//TODO: Can replace these two lines with a break.
				connections.insert(it, connectionIndex);
				return true;
			}
		}
		//If the end of the list is reached, the new connection is added to the end of the list.
		connections.insert(it, connectionIndex);
		return true;
	}

	//Checks a vector of booleans of cells that are able to be updated to see if it can update.
	bool neuralNetwork::cell::canUpdate(const std::vector<bool> &updateVec)
	{
		bool output = true;
		for (int currentConnection : connections)
		{
#if SAFE_CELL
			if (currentConnection >= (int)updateVec.size())
			{
				throw std::out_of_range("Cell contains a connection index outside the range of the provided vector.");
			}
#endif
			if (!updateVec[currentConnection])
			{
				output = false;
				break;
			}
		}
		return output;
	}

	//Copies the index of all the connections to this cell to an inputted list.
	void neuralNetwork::cell::getConnections(std::list<int> &output)
	{
		output = connections;
	}

	int neuralNetwork::cell::getIndex()
	{
		return cellIndex;
	}

	//Outputs whether this cell will backpropagate the error further.
	bool neuralNetwork::cell::getPropagateFurther()
	{
		return backPropagateFurther;
	}

	/*Attempts to remove a connection with the given index. If it removes something, this function
	  will return true. Otherwise, this function will always be false.*/
	bool neuralNetwork::cell::removeConnection(int connectionIndex)
	{
		//Checks if the inputted integer is a possible index.
#if SAFE_CELL
		if (connectionIndex < 0)
		{
			throw std::out_of_range("A negative index isn't valid.");
		}
#endif

		//Iterates over the sorted list of connections to find a connection with the given index.
		std::list<int>::iterator it = connections.begin();
		for (; it != connections.end(); ++it)
		{
			if (*it < connectionIndex)
			{
				continue;
			}
			else if (*it == connectionIndex)
			{
				connections.erase(it);
				return true;
			}
			else
			{
				break;
			}
		}
		return false;
	}

	//Sets the boolean on whether the cell will backpropagate the error further.
	void neuralNetwork::cell::setPropagateFurther(bool propFurther)
	{
		backPropagateFurther = propFurther;
	}

	//neuron:
	neuralNetwork::neuron::neuron(bool propFurther, int newIndex) :cell(propFurther, newIndex), dropRatePercent(DEFAULT_DROP_OFF_RATE),
		learningRate(DEFAULT_LEARNING_RATE), momentum(DEFAULT_MOMENTUM), previousBiasChange(0), weightDecay(DEFAULT_WEIGHT_DECAY)
	{
		bias = DEFAULT_MIN_START_WEIGHT + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (DEFAULT_MAX_START_WEIGHT - DEFAULT_MIN_START_WEIGHT)));
		actFunc = buildActFuncBundle(DEFAULT_ACTIVATION_FUNCTION);
	}

	bool neuralNetwork::neuron::addConnection(int connectionIndex)
	{
		return addConnection(connectionIndex, DEFAULT_MIN_START_WEIGHT + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (DEFAULT_MAX_START_WEIGHT - DEFAULT_MIN_START_WEIGHT))));
	}

	bool neuralNetwork::neuron::addConnection(int connectionIndex, float connectionWeight)
	{
		//Checks if the inputted index is a possible index 
#if SAFE_CELL
		if (connectionIndex < 0)
		{
			throw std::out_of_range("A negative index isn't valid.");
		}
#endif

		/*Iterates over the list of connections to see if this connection already exists. Otherwise,
		  when it finds a connection with a higher index, the new connection is added.*/
		std::list<int>::iterator connectIt = connections.begin();
		std::list<float>::iterator weightIt = connectionWeights.begin();
		std::list<float>::iterator preWeightIt = previousWeightChange.begin();

#if SAFE_CELL
		for (; connectIt != connections.end() && weightIt != connectionWeights.end() && preWeightIt != previousWeightChange.end(); ++connectIt, ++weightIt, ++preWeightIt)
#else
		for (; connectIt != connections.end(); ++connectIt, ++weightIt)
#endif

		{
			if (*connectIt == connectionIndex)
			{
				return false;
			}
			if (*connectIt > connectionIndex)
			{
				//TODO: Can replace these three lines with a break.
				connections.insert(connectIt, connectionIndex);
				connectionWeights.insert(weightIt, connectionWeight);
				previousWeightChange.insert(preWeightIt, 0.0f);
				return true;
			}
		}
		//Checks to make sure the loop wasn't broken because one list was a different length then another.
#if SAFE_CELL
		if (connectIt != connections.end() ^ weightIt != connectionWeights.end() ^ preWeightIt != previousWeightChange.end())
		{
			throw lists_not_same_length();
		}
#endif

		//If the end of the list is reached, the new connection is added to the end of the list.
		connections.insert(connectIt, connectionIndex);
		connectionWeights.insert(weightIt, connectionWeight);
		previousWeightChange.insert(preWeightIt, 0.0f);
		return true;
	}

	void neuralNetwork::neuron::backwardPropagate(std::list<std::vector<float>> &batchInput, int batchSize, std::list<std::vector<float>> &errorList, std::mutex &errorLock)
	{
#if SAFE_CELL
		//If the batch size provided isn't possible, an exception is thrown.
		if (batchSize < 1)
		{
			throw std::out_of_range("Batch size must be greater then zero.");
		}
		//Also checks that this neuron's cell index is within the range of batchInput length.
		if (cellIndex >= (int)batchInput.size())
		{
			throw std::out_of_range("Provided list of all cell batch values isn't large enough to include the current cell's index.");
		}
		//Also checks that this neuron's cell index is within the range of the errorList length.
		if (cellIndex >= (int)errorList.size())
		{
			throw std::out_of_range("Provided error list is too small to contain an index for this cell.");
		}
		//Also checks if the list of cell values and cell errors are different sizes.
		if (batchInput.size() != (int)errorList.size())
		{
			throw lists_not_same_length();
		}
		//Also checks that weights and previous change lists are the same length.
		if (connectionWeights.size() != previousWeightChange.size())
		{
			throw lists_not_same_length();
		}
		//Also checks that connections and connection weights are the same length.
		if (connections.size() != connectionWeights.size())
		{
			throw lists_not_same_length();
		}
#endif

		//If there are no connections, the error for the current cell is reset to 0.
		if (connections.size() == 0)
		{
			std::list<std::vector<float>>::iterator errorIt = errorList.begin();
			for (int currentIndex = 0; currentIndex < cellIndex; ++currentIndex, ++errorIt)
			{
			}
			for (std::vector<float>::iterator currentErrorIt = errorIt->begin(); currentErrorIt != errorIt->end(); ++currentErrorIt)
			{
				*currentErrorIt = 0;
			}
		}
		else
		{
			//Finds the error and values for the current cell.
			std::list<std::vector<float>>::iterator errorIt = errorList.begin();
			std::list<std::vector<float>>::iterator currentCellValueIt = batchInput.begin();
			if (actFunc.gradientInTermsOfFunc)
			{
				for (int currentIndex = 0; currentIndex < cellIndex; ++currentIndex, ++errorIt, ++currentCellValueIt)
				{
				}
			}
			else
			{
				for (int currentIndex = 0; currentIndex < cellIndex; ++currentIndex, ++errorIt)
				{
				}
			}

			//Updates the bias.
			previousBiasChange *= momentum;
			previousBiasChange += learningRate * (std::accumulate(errorIt->begin(), errorIt->end(), 0.0f) / batchSize);
			previousBiasChange -= weightDecay * bias;
			bias += previousBiasChange;

			//Backpropagate the error and update that weight.
			//TODO: check if copying the error improves performance.
			std::list<int>::iterator currentSearchIndex = connections.begin();
			std::list<float>::iterator currentSearchWeight = connectionWeights.begin();
			std::list<float>::iterator currentSearchPrevWeight = previousWeightChange.begin();
			std::list<std::vector<float>>::iterator errorBackIt = errorList.begin();
			std::list<std::vector<float>>::iterator valueBackIt = batchInput.begin();
			std::vector<float>::iterator currentError = errorIt->begin();
			std::vector<float>::iterator currentValue = valueBackIt->begin();
			std::vector<float>::iterator connectionError = errorBackIt->begin();
			std::vector<float>::iterator currentCellValue;

			float averageError = 0.0f;
			int currentIndex = 0;
			for (; currentSearchIndex != connections.end(); ++currentSearchIndex, ++currentSearchWeight, ++currentSearchPrevWeight)
			{
				//Iterates the value and error lists to the right connection index.
				for (; currentIndex < *currentSearchIndex && errorBackIt != errorList.end(); ++currentIndex, ++errorBackIt, ++valueBackIt)
				{
				}
#if SAFE_CELL
				//Double checks that the value of the connected cell has the same length as the current.
				if (errorIt->size() != valueBackIt->size())
				{
					throw lists_not_same_length();
				}
#endif
				//Backpropagates the error while updating the weights.
				currentError = errorIt->begin();
				currentValue = valueBackIt->begin();
				connectionError = errorBackIt->begin();
				averageError = 0.0f;

				if (actFunc.gradientInTermsOfFunc)
				{
					currentCellValue = currentCellValueIt->begin();
				}
				else
				{
#if SAFE_CELL
					//Double checks that the rawValue vector is the correct size.
					if (rawValues.size() != valueBackIt->size())
					{
						throw lists_not_same_length();
					}
#endif
					currentCellValue = rawValues.begin();
				}

				if (backPropagateFurther)
				{
#if SAFE_CELL
					//Checks that the error list of the connected has the same length as the current.
					if (errorIt->size() != errorBackIt->size())
					{
						throw lists_not_same_length();
					}
#endif
					errorLock.lock();
					for (; currentError != errorIt->end(); ++currentError, ++currentValue, ++connectionError, ++currentCellValue)
					{
						std::cout << *connectionError << " " << *currentError << " " << *currentSearchWeight << " " << *currentCellValue << std::endl;
						*connectionError += *currentError * *currentSearchWeight * actFunc.activationFunctionGradient(*currentCellValue);
						averageError += *currentError * *currentValue * actFunc.activationFunctionGradient(*currentCellValue);
					}
					errorLock.unlock();

					averageError /= batchSize;

					*currentSearchPrevWeight *= momentum;
					*currentSearchPrevWeight += learningRate * averageError;
					*currentSearchPrevWeight -= *currentSearchWeight * weightDecay;
					*currentSearchWeight += *currentSearchPrevWeight;
				}

				//If the error doesn't need to be backprop further, the value is used to update the weights.
				else
				{
					for (; currentError != errorIt->end(); ++currentError, ++currentValue, ++connectionError, ++currentCellValue)
					{
						averageError += *currentError * *currentValue * actFunc.activationFunctionGradient(*currentCellValue);
					}

					averageError /= batchSize;

					*currentSearchPrevWeight *= momentum;
					*currentSearchPrevWeight += learningRate * averageError;
					*currentSearchPrevWeight -= *currentSearchWeight * weightDecay;
					*currentSearchWeight += *currentSearchPrevWeight;
				}

			}

			//At the end, sets the error of the current neuron back to 0.
			for (std::vector<float>::iterator erIt = errorIt->begin(); erIt != errorIt->end(); ++erIt)
			{
				*erIt = 0.0f;
			}

		}
	}

	void neuralNetwork::neuron::copy(cell *&target)
	{
		//TODO: Add an exception if a non-null pointer is given.
		if (!target)
		{

		}
		else
		{
			target = new neuron(*this);
		}
	}

	void neuralNetwork::neuron::forwardPropagate(std::list<std::vector<float>> &batchInput, int batchSize)
	{
#if SAFE_CELL
		//If the batch size provided isn't possible, an exception is thrown.
		if (batchSize < 1)
		{
			throw std::out_of_range("Batch size must be greater then zero.");
		}
		//Also checks that this neuron's cell index is within the range of batchInput length.
		if (cellIndex >= (int)batchInput.size())
		{
			throw std::out_of_range("Provided list of all cell batch values isn't large enough to include the current cell's index.");
		}
#endif

		std::vector<float> cellBatchValues(batchSize, bias);
		std::vector<float>::iterator valueIt, currentCellIt;
		std::list<std::vector<float>>::iterator cellValueIt = batchInput.begin();
		std::list<int>::iterator currentSearchIndex = connections.begin();
		std::list<float>::iterator currentSearchWeight = connectionWeights.begin();

		//Iterates through the cell values while keeping tracking of the current index.
		for (int currentCell = 0; cellValueIt != batchInput.end(); ++currentCell, ++cellValueIt)
		{
			//Once an index that has a connection is reached. Each value is added to the current batch value after
			//being multiplied by the weight of the connection.
			if (currentCell == *currentSearchIndex)
			{
				addVectors(cellBatchValues, *cellValueIt, *currentSearchWeight);
				//Iterates the list of connections to the next search.
				++currentSearchIndex;
				++currentSearchWeight;
			}

			//If there isn't anymore connections that need to be searched for, the loop is broken.
#if SAFE_CELL
			if (currentSearchIndex == connections.end() || currentSearchWeight == connectionWeights.end())
			{
				//Checks to make sure they are both at the end. Otherwise, an exception is thrown.
				if (currentSearchIndex == connections.end() ^ currentSearchWeight == connectionWeights.end())
				{
					throw lists_not_same_length();
				}
				else
				{
					break;
				}
			}
#else
			if (currentSearchIndex == connections.end())
			{
				break;
			}
#endif
		}
#if SAFE_CELL
		//If there are some index connections that haven't been reach, an exception is thrown.
		if (currentSearchIndex != connections.end())
		{
			throw std::out_of_range("Provide list of batch values of each index was too short.");
		}
#endif

		//Goes through and applies the activation function and save the raw values before applying the activation function.
		if (!actFunc.gradientInTermsOfFunc)
		{
			rawValues = cellBatchValues;
		}
		for (valueIt = cellBatchValues.begin(); valueIt != cellBatchValues.end(); ++valueIt)
		{
			*valueIt = actFunc.activationFunction(*valueIt);
		}

		//If dropoff, randomly sets some of the batch values to zero based on percent.
		//TODO: Can rewrite this function where I figure out which values are going to be zero and not calculate them for performance.
		if (dropRatePercent > 0)
		{
			for (valueIt = cellBatchValues.begin(); valueIt != cellBatchValues.end(); ++valueIt)
			{
				if (dropRatePercent > static_cast <float> (rand()) / static_cast <float> (RAND_MAX))
				{
					*valueIt = 0;
				}
			}
		}

		//Updates the value in the list of all cells for this cell.
		cellValueIt = batchInput.begin();
		for (int currentCellIndex = 0; currentCellIndex < cellIndex; ++currentCellIndex, ++cellValueIt)
		{
		}


		*cellValueIt = cellBatchValues;
	}

	activationFunctionInfo neuralNetwork::neuron::getActivationFunction()
	{
		return actFunc;
	}

	float neuralNetwork::neuron::getBias()
	{
		return bias;
	}

	float neuralNetwork::neuron::getDropRatePercent()
	{
		return dropRatePercent;
	}

	float neuralNetwork::neuron::getLearningRate()
	{
		return learningRate;
	}

	float neuralNetwork::neuron::getMomentum()
	{
		return momentum;
	}

	float neuralNetwork::neuron::getPreviousBiasChange()
	{
		return previousBiasChange;
	}

	void neuralNetwork::neuron::getPreviousWeightChanges(std::list<float> &output)
	{
		output = previousWeightChange;
	}

	float neuralNetwork::neuron::getWeightDecay()
	{
		return weightDecay;
	}

	void neuralNetwork::neuron::getWeights(std::list<float> &output)
	{
		output = connectionWeights;
	}

	bool neuralNetwork::neuron::removeConnection(int connectionIndex)
	{

#if SAFE_CELL
		//Checks if the inputted integer is a possible index.
		if (connectionIndex < 0)
		{
			throw std::out_of_range("A negative index isn't valid.");
		}
#endif

		//Iterates over the sorted list of connections to find a connection with the given index.
		std::list<int>::iterator connectIt = connections.begin();
		std::list<float>::iterator weightIt = connectionWeights.begin();
		std::list<float>::iterator preWeightIt = previousWeightChange.begin();

#if SAFE_CELL
		for (; connectIt != connections.end() && weightIt != connectionWeights.end() && preWeightIt != previousWeightChange.end(); ++connectIt, ++weightIt, ++preWeightIt)
#else
		for (; connectIt != connections.end(); ++connectIt, ++weightIt, ++preWeightIt)
#endif
		{
			if (*connectIt < connectionIndex)
			{
				continue;
			}
			//If the connection is found, it's erased from all lists.
			else if (*connectIt == connectionIndex)
			{
				connections.erase(connectIt);
				connectionWeights.erase(weightIt);
				previousWeightChange.erase(preWeightIt);
				return true;
			}
			else
			{
				break;
			}
		}

#if SAFE_CELL
		//Checks to make sure the loop wasn't broken because one list was a different length then another.
		//TODO: This logic for checking whether all 3 lists are the same length can be simplified.
		if (connectIt != connections.end())
		{
			if (weightIt == connectionWeights.end() || preWeightIt == previousWeightChange.end())
			{
				throw lists_not_same_length();
			}
		}
		else
		{
			if (weightIt != connectionWeights.end() || preWeightIt != previousWeightChange.end())
			{
				throw lists_not_same_length();
			}
		}
#endif

		return false;
	}

	void neuralNetwork::neuron::setBias(float newBias)
	{
		bias = newBias;
	}

	void neuralNetwork::neuron::setDropRatePercent(float newDropRate)
	{
#if SAFE_CELL
		if (newDropRate > 1.0f || newDropRate < 0.0f)
		{
			throw std::out_of_range("The drop rate percentage cannot be changed to value less then zero or greater then one.");
		}
#endif
		dropRatePercent = newDropRate;
	}

	void neuralNetwork::neuron::setLearningRate(float newLearningRate)
	{
#if SAFE_CELL
		if (newLearningRate < 0.0f)
		{
			throw std::out_of_range("The learning rate cannot be changed to value less then zero.");
		}
#endif
		learningRate = newLearningRate;
	}

	void neuralNetwork::neuron::setMomentum(float newMomentum)
	{
#if SAFE_CELL
		if (newMomentum < 0.0f)
		{
			throw std::out_of_range("The momentum cannot be changed to value less then zero.");
		}
#endif
		momentum = newMomentum;
	}

	void neuralNetwork::neuron::setPreviousBiasChange(float newBiasChange)
	{
		previousBiasChange = newBiasChange;
	}

	void neuralNetwork::neuron::setPreviousWeightChanges(std::list<float> &ref)
	{
#if SAFE_CELL
		if (ref.size() != previousWeightChange.size())
		{
			throw lists_not_same_length();
		}
#endif
		previousWeightChange = ref;
	}

	void neuralNetwork::neuron::setWeightDecay(float newWeightDecay)
	{
#if SAFE_CELL
		if (newWeightDecay < 0.0f)
		{
			throw std::out_of_range("The weight decay cannot be changed to value less then zero.");
		}
#endif
		weightDecay = newWeightDecay;
	}

	void neuralNetwork::neuron::setWeights(std::list<float> &ref)
	{
#if SAFE_CELL
		if (ref.size() != connectionWeights.size())
		{
			throw lists_not_same_length();
		}
#endif
		connectionWeights = ref;
	}

	//neuralNetwork:
	neuralNetwork::neuralNetwork():inputNodes(0), outputNodes(0)
	{

	}

	neuralNetwork::neuralNetwork(const neuralNetwork &ref) : inputNodes(ref.inputNodes), outputNodes(ref.outputNodes)
	{
		cell *tempCell = NULL;
		for (std::list<std::list<cell*>>::const_iterator scheduleIt = ref.schedule.begin(); scheduleIt != ref.schedule.end(); ++scheduleIt)
		{
			schedule.push_back(std::list<cell*>());
			for (std::list<cell*>::const_iterator it = scheduleIt->begin(); it != scheduleIt->end(); ++it)
			{
				(*it)->copy(tempCell);
				schedule.back().push_back(tempCell);
				tempCell = NULL;
			}
		}
	}

	neuralNetwork::~neuralNetwork()
	{
		inputNodes = 0;
		outputNodes = 0;
		for (std::list<std::list<cell*>>::iterator scheduleIt = schedule.begin(); scheduleIt != schedule.end(); ++scheduleIt)
		{
			for (std::list<cell*>::iterator it = scheduleIt->begin(); it != scheduleIt->end(); ++it)
			{
				delete *it;
			}
		}
		schedule.clear();
	}

	neuralNetwork& neuralNetwork::operator=(const neuralNetwork &ref)
	{
		cell *tempCell = NULL;
		if (this != &ref)
		{
			inputNodes = ref.inputNodes;
			outputNodes = ref.outputNodes;

			//TODO: Could resize the list to match the reference and clear the list before copying.
			//Deletes the schedule and creates a copy of the list.
			for (std::list<std::list<cell*>>::iterator scheduleIt = schedule.begin(); scheduleIt != schedule.end(); ++scheduleIt)
			{
				for (std::list<cell*>::iterator it = scheduleIt->begin(); it != scheduleIt->end(); ++it)
				{
					delete *it;
				}
			}
			schedule.clear();

			for (std::list<std::list<cell*>>::const_iterator scheduleIt = ref.schedule.begin(); scheduleIt != ref.schedule.end(); ++scheduleIt)
			{
				schedule.push_back(std::list<cell*>());
				for (std::list<cell*>::const_iterator it = scheduleIt->begin(); it != scheduleIt->end(); ++it)
				{
					(*it)->copy(tempCell);
					schedule.back().push_back(tempCell);
					tempCell = NULL;
				}
			}
		}
		return *this;
	}
}