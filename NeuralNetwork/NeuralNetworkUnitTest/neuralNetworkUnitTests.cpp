//Copyright(C) 2020 "Daniel Bramblett" <daniel.r.bramblett@gmail.com>
#include "stdafx.h"
#include "CppUnitTest.h"
#include "testNeuralNetwork.h"
#include "testHelperFunctions.h"
#include "../NeuralNetwork/neuralNetwork.cpp"
#include "../NeuralNetwork/activationFunctions.cpp"
#include "../NeuralNetwork/helperFunctions.cpp"

#include<list>
#include<vector>
#include<string>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace NeuralNetwork;

//TODO: Refactor to move variables to fields to prevent initialize new variables for each test.
/*TODO: Figure out how to split each class into its own file to shorten file lengths.
 Link the .obj instead of including the .cpp without creating a path dependent on the computer.
*/

namespace NeuralNetworkUnitTest
{	
	static float FLOAT_TEST_RANGE = 0.0001f;

	TEST_CLASS(cellUnitTests)
	{
	public:
		
		//Tests that the cell constructor correctly initializes the fields.
		TEST_METHOD(cellConstructer)
		{
			testNeuralNetwork::testCell x(true, 0), y(false, 3);
			std::list<int> connectedIndexes;
			Assert::IsTrue(x.getPropagateFurther());
			Assert::IsFalse(y.getPropagateFurther());
			x.getConnections(connectedIndexes);
			Assert::AreEqual(int(connectedIndexes.size()), 0);
			Assert::AreEqual(x.getIndex(), 0);
			y.getConnections(connectedIndexes);
			Assert::AreEqual(int(connectedIndexes.size()), 0);
			Assert::AreEqual(y.getIndex(), 3);
		}

		//Tests the addConnection and removeConnection methods.
		TEST_METHOD(addRemoveConnection)
		{
			testNeuralNetwork::testCell x(true, 0);
			std::list<int> connectedIndexes;
			int testIndexes[] = { 1, 4 ,6 };
			int count = 0;

			//Checks to make sure the number of connections is zero before starting.
			x.getConnections(connectedIndexes);
			Assert::AreEqual(int(connectedIndexes.size()), 0);

			//Checks to make sure that three connections are added in the correct order.
			//Also checks to make that duplicates are rejected.
			Assert::IsTrue(x.addConnection(testIndexes[1]));
			Assert::IsTrue(x.addConnection(testIndexes[2]));
			Assert::IsTrue(x.addConnection(testIndexes[0]));
			Assert::IsFalse(x.addConnection(testIndexes[1]));
			x.getConnections(connectedIndexes);
			Assert::AreEqual(int(connectedIndexes.size()), 3);
			for (std::list<int>::iterator it = connectedIndexes.begin(); it != connectedIndexes.end(); ++it)
			{
				Assert::AreEqual(*it, testIndexes[count++]);
			}

			//Checks to make sure the connections are removed correctly from middle, largest, smallest connection index.
			//Also checks to make sure it doesn't remove anything if an index isn't found.
			count = 0;
			Assert::IsTrue(x.removeConnection(testIndexes[1]));
			x.getConnections(connectedIndexes);
			Assert::AreEqual(int(connectedIndexes.size()), 2);
			for (std::list<int>::iterator it = connectedIndexes.begin(); it != connectedIndexes.end(); ++it)
			{
				Assert::AreEqual(*it, testIndexes[count]);
				count += 2;
			}
			Assert::IsFalse(x.removeConnection(testIndexes[1]));
			Assert::IsTrue(x.removeConnection(testIndexes[2]));
			x.getConnections(connectedIndexes);
			Assert::AreEqual(int(connectedIndexes.size()), 1);
			Assert::AreEqual(*connectedIndexes.begin(), testIndexes[0]);
			Assert::IsTrue(x.removeConnection(testIndexes[0]));
			x.getConnections(connectedIndexes);
			Assert::AreEqual(int(connectedIndexes.size()), 0);
		}

		TEST_METHOD(canUpdate)
		{
			testNeuralNetwork::testCell x(true, 0);
			std::vector<bool> testVector;
			testVector.resize(3);
			testVector[0] = false;
			testVector[1] = false;
			testVector[2] = false;

			//Cells without connections should always return true.
			Assert::IsTrue(x.canUpdate(testVector));

			//Checks to make sure that only connected cells are checked and that the method
			//performs correctly in each case.
			Assert::IsTrue(x.addConnection(0));
			Assert::IsFalse(x.canUpdate(testVector));
			testVector[0] = true;
			Assert::IsTrue(x.canUpdate(testVector));

			Assert::IsTrue(x.addConnection(2));
			Assert::IsFalse(x.canUpdate(testVector));
			testVector[2] = true;
			Assert::IsTrue(x.canUpdate(testVector));

			Assert::IsTrue(x.addConnection(1));
			Assert::IsFalse(x.canUpdate(testVector));
			testVector[1] = true;
			Assert::IsTrue(x.canUpdate(testVector));
		}

		//Checks to make sure the correct exception is thrown if the checks, that are only run
		//when the SAFE_CELL precompilation flag is on, fail.
		TEST_METHOD(safeCellChecks)
		{
#if SAFE_CELL
			testNeuralNetwork::testCell x(true, 0);
			std::vector<bool> testVec;

			//Tests that a negative index can't be assigned to a cell.
			Assert::ExpectException<std::out_of_range>([&] {testNeuralNetwork::testCell y(true, -1); });

			//Tests that a negative connection index is not added.
			Assert::ExpectException<std::out_of_range>([&]{x.addConnection(-4); });
			Assert::IsTrue(x.addConnection(4));

			//Tests that an exception is thrown if there is a connection index larger or equal the size of the vector.
			Assert::ExpectException<std::out_of_range>([&] {x.canUpdate(testVec); });
			Assert::IsTrue(x.removeConnection(4));
			Assert::IsTrue(x.addConnection(0));
			Assert::ExpectException<std::out_of_range>([&] {x.canUpdate(testVec); });

			//Tests that a negative index isn't searched for when trying to remove a connection.
			Assert::ExpectException<std::out_of_range>([&] {x.removeConnection(-4); });

#else
			//If the precompilation flag is off, this test always passes.
			Assert::IsTrue(true);
#endif
		}

		//Checks to make sure the setPropagateFurther method correctly set the field.
		TEST_METHOD(setPropagateFurther)
		{
			testNeuralNetwork::testCell x(true, 0);
			Assert::IsTrue(x.getPropagateFurther());
			x.setPropagateFurther(false);
			Assert::IsFalse(x.getPropagateFurther());
			x.setPropagateFurther(true);
			Assert::IsTrue(x.getPropagateFurther());
		}
	};

	TEST_CLASS(neuronUnitTests)
	{
	public:

		//Tests that the neuron class constructor correctly initializes the fields.
		TEST_METHOD(neuronConstructer)
		{
			testNeuralNetwork::testNeuron a(true, 0), b(false, 2);
			std::list<float> testList;
			float testPoints[] = { -1.0f, 0.0f, 1.0f };
			activationFunctionInfo testAct = buildActFuncBundle(DEFAULT_ACTIVATION_FUNCTION);

			//Checks the cell class constructor is correctly called.
			Assert::IsTrue(a.getPropagateFurther());
			Assert::IsFalse(b.getPropagateFurther());
			Assert::AreEqual(a.getIndex(), 0);
			Assert::AreEqual(b.getIndex(), 2);

			//Checks the fields in the neuron class.
			Assert::IsTrue(a.getBias() >= DEFAULT_MIN_START_WEIGHT);
			Assert::IsTrue(a.getBias() <= DEFAULT_MAX_START_WEIGHT);
			Assert::AreEqual(a.getDropRatePercent(), DEFAULT_DROP_OFF_RATE);
			Assert::AreEqual(a.getLearningRate(), DEFAULT_LEARNING_RATE);
			Assert::AreEqual(a.getMomentum(), DEFAULT_MOMENTUM);
			Assert::AreEqual(a.getPreviousBiasChange(), 0.0f);
			Assert::AreEqual(a.getWeightDecay(), DEFAULT_WEIGHT_DECAY);
			a.getWeights(testList);
			Assert::AreEqual((int)testList.size(), 0);
			a.getPreviousWeightChanges(testList);
			Assert::AreEqual((int)testList.size(), 0);

			Assert::IsTrue(a.getActivationFunction().gradientInTermsOfFunc);
			for (float currentPoint : testPoints)
			{
				Assert::AreEqual(a.getActivationFunction().activationFunction(currentPoint), testAct.activationFunction(currentPoint));
			}
		}
		//Tests that the addConnection() and removeConnection() function correctly.
		TEST_METHOD(addRemoveConnection)
		{
			testNeuralNetwork::testNeuron a(true, 0);
			std::list<float> testWeightList, testPreviousWeightList;
			std::list<int> testIndexList;
			int sortedOrder[] = { 1, 0, 2 };
			int testIndexes[] = { 1, 0, 2 };
			float testWeights[] = { 0.0f, -1.0f, 1.0f };
			bool assignWeight[] = { false, true, true };

			//Checks the default neuron has empty lists for connections, weights, and previous weights.
			a.getConnections(testIndexList);
			a.getWeights(testWeightList);
			a.getPreviousWeightChanges(testPreviousWeightList);
			Assert::AreEqual((int)testIndexList.size(), 0);
			Assert::AreEqual((int)testWeightList.size(), 0);
			Assert::AreEqual((int)testPreviousWeightList.size(), 0);

			//Adds three connections. Two have assigned weight values and one is randomly generated.
			for (int i = 0; i < 3; ++i)
			{
				if (assignWeight[i])
				{
					Assert::IsTrue(a.addConnection(testIndexes[i], testWeights[i]));
				}
				else
				{
					Assert::IsTrue(a.addConnection(testIndexes[i]));
				}
			}
			a.getConnections(testIndexList);
			a.getWeights(testWeightList);
			a.getPreviousWeightChanges(testPreviousWeightList);
			Assert::AreEqual((int)testIndexList.size(), 3);
			Assert::AreEqual((int)testWeightList.size(), 3);
			Assert::AreEqual((int)testPreviousWeightList.size(), 3);

			//Makes sure it returns false and doesn't change the lists with when a dupe index is attempted to be added.
			for (int i = 0; i < 3; ++i)
			{
				if (assignWeight[i])
				{
					Assert::IsFalse(a.addConnection(testIndexes[i], testWeights[i]));
				}
				else
				{
					Assert::IsFalse(a.addConnection(testIndexes[i]));
				}
			}
			a.getConnections(testIndexList);
			a.getWeights(testWeightList);
			a.getPreviousWeightChanges(testPreviousWeightList);
			Assert::AreEqual((int)testIndexList.size(), 3);
			Assert::AreEqual((int)testWeightList.size(), 3);
			Assert::AreEqual((int)testPreviousWeightList.size(), 3);

			//Checks all three of the lists to make sure the values are in the correct spots.
			std::list<int>::iterator indexIt = testIndexList.begin();
			int count = 0;
			for (std::list<float>::iterator weightIt = testWeightList.begin(), preWeightIt = testPreviousWeightList.begin();
				weightIt != testWeightList.end(); ++indexIt, ++weightIt, ++preWeightIt, ++count)
			{
				Assert::AreEqual(*indexIt, testIndexes[sortedOrder[count]]);
				if (assignWeight[sortedOrder[count]])
				{
					Assert::AreEqual(*weightIt, testWeights[sortedOrder[count]]);
				}
				else
				{
					Assert::IsTrue(*weightIt >= DEFAULT_MIN_START_WEIGHT);
					Assert::IsTrue(*weightIt <= DEFAULT_MAX_START_WEIGHT);
				}
				Assert::AreEqual(*preWeightIt, 0.0f);
			}

			//Removes the last test value.
			Assert::IsTrue(a.removeConnection(testIndexes[--count]));
			//Makes sure that trying to remove the same index again fails.
			Assert::IsFalse(a.removeConnection(testIndexes[count]));

			//Checks the lists to make sure that everything is in the correct order.
			a.getConnections(testIndexList);
			a.getWeights(testWeightList);
			a.getPreviousWeightChanges(testPreviousWeightList);
			Assert::AreEqual((int)testIndexList.size(), 2);
			Assert::AreEqual((int)testWeightList.size(), 2);
			Assert::AreEqual((int)testPreviousWeightList.size(), 2);
			indexIt = testIndexList.begin();
			count = 0;
			for (std::list<float>::iterator weightIt = testWeightList.begin(), preWeightIt = testPreviousWeightList.begin();
				weightIt != testWeightList.end(); ++indexIt, ++weightIt, ++preWeightIt, ++count)
			{
				Assert::AreEqual(*indexIt, testIndexes[sortedOrder[count]]);
				if (assignWeight[sortedOrder[count]])
				{
					Assert::AreEqual(*weightIt, testWeights[sortedOrder[count]]);
				}
				else
				{
					Assert::IsTrue(*weightIt >= DEFAULT_MIN_START_WEIGHT);
					Assert::IsTrue(*weightIt <= DEFAULT_MAX_START_WEIGHT);
				}
				Assert::AreEqual(*preWeightIt, 0.0f);
			}

			//Removes the first test index.
			Assert::IsTrue(a.removeConnection(testIndexes[0]));
			Assert::IsFalse(a.removeConnection(testIndexes[0]));
			a.getConnections(testIndexList);
			a.getWeights(testWeightList);
			a.getPreviousWeightChanges(testPreviousWeightList);
			Assert::AreEqual((int)testIndexList.size(), 1);
			Assert::AreEqual((int)testWeightList.size(), 1);
			Assert::AreEqual((int)testPreviousWeightList.size(), 1);
			indexIt = testIndexList.begin();
			count = 1;
			for (std::list<float>::iterator weightIt = testWeightList.begin(), preWeightIt = testPreviousWeightList.begin();
				weightIt != testWeightList.end(); ++indexIt, ++weightIt, ++preWeightIt, ++count)
			{
				Assert::AreEqual(*indexIt,testIndexes[count]);
				if (assignWeight[count])
				{
					Assert::AreEqual(*weightIt, testWeights[count]);
				}
				else
				{
					Assert::IsTrue(*weightIt >= DEFAULT_MIN_START_WEIGHT);
					Assert::IsTrue(*weightIt <= DEFAULT_MAX_START_WEIGHT);
				}
				Assert::AreEqual(*preWeightIt, 0.0f);
			}

			//Removes the last element
			Assert::IsTrue(a.removeConnection(testIndexes[1]));
			Assert::IsFalse(a.removeConnection(testIndexes[1]));
			a.getConnections(testIndexList);
			a.getWeights(testWeightList);
			a.getPreviousWeightChanges(testPreviousWeightList);
			Assert::AreEqual((int)testIndexList.size(), 0);
			Assert::AreEqual((int)testWeightList.size(), 0);
			Assert::AreEqual((int)testPreviousWeightList.size(), 0);
		}
		/*Tests the backPropagation function when further error propagation is needed and the batchsize
		 *is greater then one. */
		TEST_METHOD(backwardPropagation)
		{
			testNeuralNetwork::testNeuron neuronTest(true, 2);
			neuronTest.addConnection(0, 0.4f);
			neuronTest.addConnection(3, 0.7f);
			neuronTest.setBias(0.6f);
			std::list<std::vector<float>> testValues;
			std::list<std::vector<float>> testError;
			std::list<float> testList;
			std::mutex testMutex;
			std::vector<float> neuronError(2);
			neuronError[0] = -0.4f;
			neuronError[1] = 0.5f;
			for (int i = 0; i < 6; ++i)
			{
				std::vector<float> temp(2);
				temp[0] = 0.1f * (1.0f + i);
				temp[1] = 0.15f * (1.0f + i);
				testValues.push_back(temp);
				if (i == 2)
				{
					testError.push_back(neuronError);
				}
				else
				{
					testError.push_back(std::vector<float>(2, 0.0f));
				}
			}

			neuronTest.backwardPropagate(testValues, 2, testError, testMutex);

			//Checks to make sure the values are not changed.
			std::list<std::vector<float>>::iterator valueIt = testValues.begin();
			for (int i = 0; i < 6; ++i, ++valueIt)
			{
				Assert::AreEqual((*valueIt)[0], 0.1f * (1.0f + i));
				Assert::AreEqual((*valueIt)[1], 0.15f * (1.0f + i));
			}

			//Checks to make sure the error was probably backpropagated.
			std::list<std::vector<float>>::iterator errorIt = testError.begin();
			for (int i = 0; i < 6; ++i, ++errorIt)
			{
				if (i == 0)
				{
					Assert::IsTrue(floatInBounds((*errorIt)[0], -0.0336f, FLOAT_TEST_RANGE));
					Assert::IsTrue(floatInBounds((*errorIt)[1], 0.0495f, FLOAT_TEST_RANGE));
				}
				else if (i == 3)
				{
					Assert::IsTrue(floatInBounds((*errorIt)[0], -0.0588f, FLOAT_TEST_RANGE));
					Assert::IsTrue(floatInBounds((*errorIt)[1], 0.086625f, FLOAT_TEST_RANGE));
				}
				else
				{
					for (std::vector<float>::iterator currentErrorIt = errorIt->begin(); currentErrorIt != errorIt->end(); ++currentErrorIt)
					{
						Assert::AreEqual(*currentErrorIt, 0.0f);
					}
				}
			}
			//Checks the weights are correct.
			neuronTest.getWeights(testList);
			Assert::IsTrue(floatInBounds(*testList.begin(), 0.4f + (DEFAULT_LEARNING_RATE * 0.00508125f), FLOAT_TEST_RANGE));
			Assert::IsTrue(floatInBounds(*(++testList.begin()), 0.7f + (DEFAULT_LEARNING_RATE * 0.020325f), FLOAT_TEST_RANGE));

			//Checks the previous weight changes are correct.
			neuronTest.getPreviousWeightChanges(testList);
			Assert::IsTrue(floatInBounds(*testList.begin(), DEFAULT_LEARNING_RATE * 0.00508125f, FLOAT_TEST_RANGE));
			Assert::IsTrue(floatInBounds(*(++testList.begin()), DEFAULT_LEARNING_RATE * 0.020325f, FLOAT_TEST_RANGE));
		
			//Checks the bias and the previous change were updated correctly.
			Assert::IsTrue(floatInBounds(neuronTest.getBias(), 0.6f + (DEFAULT_LEARNING_RATE * 0.05f), FLOAT_TEST_RANGE));
			Assert::IsTrue(floatInBounds(neuronTest.getPreviousBiasChange(), DEFAULT_LEARNING_RATE * 0.05f, FLOAT_TEST_RANGE));
		}

		/*Tests the backPropagation() function with a batch size of one and the error shouldn't 
		 *backpropagate further. */
		TEST_METHOD(backwardPropagationOneBatch)
		{
			testNeuralNetwork::testNeuron neuronTest(false, 2);
			neuronTest.addConnection(0, 0.4f);
			neuronTest.addConnection(3, 0.7f);
			neuronTest.setBias(0.6f);
			std::list<std::vector<float>> testValues;
			std::list<std::vector<float>> testError;
			std::list<float> testList;
			std::mutex testMutex;
			std::vector<float> neuronError(1);
			neuronError[0] = -0.4f;
			for (int i = 0; i < 6; ++i)
			{
				std::vector<float> temp(1);
				temp[0] = 0.1f * (1.0f + i);
				testValues.push_back(temp);
				if (i == 2)
				{
					testError.push_back(neuronError);
				}
				else
				{
					testError.push_back(std::vector<float>(1, 0.0f));
				}
			}

			neuronTest.backwardPropagate(testValues, 1, testError, testMutex);

			//Checks to make sure the values are not changed.
			std::list<std::vector<float>>::iterator valueIt = testValues.begin();
			for (int i = 0; i < 6; ++i, ++valueIt)
			{
				Assert::AreEqual((*valueIt)[0], 0.1f * (1.0f + i));
			}

			//Checks to make sure the error wasn't propagated.
			std::list<std::vector<float>>::iterator errorIt = testError.begin();
			for (int i = 0; i < 6; ++i, ++errorIt)
			{
				for (std::vector<float>::iterator currentErrorIt = errorIt->begin(); currentErrorIt != errorIt->end(); ++currentErrorIt)
				{
					Assert::AreEqual(*currentErrorIt, 0.0f);
				}
			}

			//Checks the weights are correct.
			neuronTest.getWeights(testList);
			Assert::IsTrue(floatInBounds(*testList.begin(), 0.4f + (DEFAULT_LEARNING_RATE * -0.0084f), FLOAT_TEST_RANGE));
			Assert::IsTrue(floatInBounds(*(++testList.begin()), 0.7f + (DEFAULT_LEARNING_RATE * -0.0336f), FLOAT_TEST_RANGE));
		
			//Checks the previous weight changes are correct.
			neuronTest.getPreviousWeightChanges(testList);
			Assert::IsTrue(floatInBounds(*testList.begin(), DEFAULT_LEARNING_RATE * -0.0084f, FLOAT_TEST_RANGE));
			Assert::IsTrue(floatInBounds(*(++testList.begin()), DEFAULT_LEARNING_RATE * -0.0336f, FLOAT_TEST_RANGE));

			//Checks the bias and the previous change were updated correctly.
			Assert::IsTrue(floatInBounds(neuronTest.getBias(), 0.6f + (DEFAULT_LEARNING_RATE * -0.4f), FLOAT_TEST_RANGE));
			Assert::IsTrue(floatInBounds(neuronTest.getPreviousBiasChange(), DEFAULT_LEARNING_RATE * -0.4f, FLOAT_TEST_RANGE));
		}
		TEST_METHOD(forwardPropagation)
		{
			//TODO: Not setting the bias so this test shouldn't pass.
			testNeuralNetwork::testNeuron neuronTest(true, 2);
			neuronTest.addConnection(0, 0.4f);
			neuronTest.addConnection(3, 0.7f);
			neuronTest.setBias(0.3f);
			std::list<std::vector<float>> testValues;
			for (int i = 0; i < 6; ++i)
			{
				std::vector<float> temp;
				temp.push_back(0.1f * (-2.0f + i));
				temp.push_back(0.15f * (-2.0f + i));
				testValues.push_back(temp);
			}
			neuronTest.forwardPropagate(testValues, 2);

			std::list<std::vector<float>>::iterator valueIt = testValues.begin();
			for (int i = 0; i < 6; ++i, ++valueIt)
			{
				if (i != 2)
				{
					Assert::AreEqual((*valueIt)[0], 0.1f * (-2.0f + i));
					Assert::AreEqual((*valueIt)[1], 0.15f * (-2.0f + i));
				}
				else
				{
					Assert::IsTrue(floatInBounds((*valueIt)[0], sigmoid(0.29f), FLOAT_TEST_RANGE));
					Assert::IsTrue(floatInBounds((*valueIt)[1], sigmoid(0.285f), FLOAT_TEST_RANGE));
				}
			}
		}

		TEST_METHOD(setBias)
		{
			testNeuralNetwork::testNeuron a(true, 0);
			a.setBias(1.42f);
			Assert::AreEqual(a.getBias(), 1.42f);
		}
		TEST_METHOD(setDropRatePercent)
		{
			testNeuralNetwork::testNeuron a(true, 0);
			a.setDropRatePercent(0.4f);
			Assert::AreEqual(a.getDropRatePercent(), 0.4f);
		}
		TEST_METHOD(setLearningRate)
		{
			testNeuralNetwork::testNeuron a(true, 0);
			a.setLearningRate(0.65f);
			Assert::AreEqual(a.getLearningRate(), 0.65f);
		}
		TEST_METHOD(setMomentum)
		{
			testNeuralNetwork::testNeuron a(true, 0);
			a.setMomentum(0.3f);
			Assert::AreEqual(a.getMomentum(), 0.3f);
		}
		TEST_METHOD(setPreviousBiasChange)
		{
			testNeuralNetwork::testNeuron a(true, 0);
			a.setPreviousBiasChange(-2.3f);
			Assert::AreEqual(a.getPreviousBiasChange(), -2.3f);
		}
		TEST_METHOD(setPreviousWeightChanges)
		{
			testNeuralNetwork::testNeuron a(true, 0);
			std::list<float> temp;
			temp.push_back(0.25f);
			a.addConnection(0);
			a.setPreviousWeightChanges(temp);
			temp.clear();
			a.getPreviousWeightChanges(temp);
			Assert::AreEqual((int)temp.size(), 1);
			Assert::AreEqual(*temp.begin(), 0.25f);
		}
		TEST_METHOD(setWeightDecay)
		{
			testNeuralNetwork::testNeuron a(true, 0);
			a.setWeightDecay(0.00001f);
			Assert::AreEqual(a.getWeightDecay(), 0.00001f);
		}

		TEST_METHOD(setWeights)
		{
			testNeuralNetwork::testNeuron a(true, 0);
			std::list<float> temp;
			temp.push_back(1.23f);
			a.addConnection(0);
			a.setWeights(temp);
			temp.clear();
			a.getWeights(temp);
			Assert::AreEqual((int)temp.size(), 1);
			Assert::AreEqual(*temp.begin(), 1.23f);
		}
	};
}