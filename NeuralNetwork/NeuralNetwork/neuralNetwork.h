//Copyright(C) 2020 "Daniel Bramblett" <daniel.r.bramblett@gmail.com>
/*Header containing the prototype for the neuralNetwork class along including the prototypes of
 *all the nested classes inside of it.*/

#ifndef NEURAL_NET_LIB
#define NEURAL_NET_LIB

#include "activationFunctions.h"
#include "preprocessorFlags.h"
#include<list>
#include<mutex>
#include<vector>

namespace NeuralNetwork
{
	//Default values used by the neuralNetwork.
	static std::string DEFAULT_ACTIVATION_FUNCTION = "sigmoid";
	static float DEFAULT_DROP_OFF_RATE = 0.0f;
	static float DEFAULT_LEARNING_RATE = 0.2f;
	static float DEFAULT_MAX_START_WEIGHT = 1.0f;
	static float DEFAULT_MIN_START_WEIGHT = -1.0f;
	static float DEFAULT_MOMENTUM = 0.9f;
	static float DEFAULT_WEIGHT_DECAY = 0.0f;

	class neuralNetwork
	{
	public:
		//TODO: Implement the neuralNetwork class.
	protected:
		/*Nested abstract cell class which represents each cell in the neural network.*/
		class cell
		{
		public:
			cell(bool, int);
			/*Attempts to add a connection with the given index. Will return false if a connection
			 *with that index already exists.*/
			virtual bool addConnection(int);
			/*Checks a vector of all indexes that can update and returns whether it can update.*/
			bool canUpdate(const std::vector<bool>&);
			void getConnections(std::list<int>&);
			int getIndex();
			bool getPropagateFurther();
			/*Attempts to remove a connection with the given index. Will return false if a connection
			 *doesn't exist with index already.*/
			virtual bool removeConnection(int);
			void setPropagateFurther(bool);

			//Pure virutal functions:
			virtual void backwardPropagate(std::list < std::vector<float>>&, int, std::list<std::vector<float>>&, std::mutex&) = 0;
			virtual void forwardPropagate(std::list < std::vector<float>>&, int) = 0;
		protected:
			//Whether the error needs to be back propagated further.
			bool backPropagateFurther;
			//Index of this cell.
			int cellIndex;
			//List of cells connections.
			std::list<int> connections;
		};
		/*Nested neuron class that represents a neuron in a neural network.*/
		//TODO: Add in final keyword and rewrite test neuralNetwork class.
		//TODO: Consider keeping pointers to the vectors of connected cell's values and error to prevent
		//looping.
		class neuron : public cell
		{
		public:
			neuron(bool, int);
			/*Attempts to add a connection given an index. Will return false if a connection to
			 *that index already exists. Also, if no weight is given, a random weight is generated
			 *for the connection.*/
			bool addConnection(int);
			bool addConnection(int, float);
			/*Backwards propagates the error of this neuron onto the cells it's connect to. Then,
			 *the weights to each connection is updated before returning the error of this cell to
			 *zero.*/
			void backwardPropagate(std::list < std::vector<float>>&, int, std::list<std::vector<float>>&, std::mutex&);
			/*Uses the values from the cells that this neuron is connected to calculate the value of 
			 *this neuron.*/
			void forwardPropagate(std::list < std::vector<float>>&, int);
			activationFunctionInfo getActivationFunction();
			float getBias();
			float getDropRatePercent();
			float getLearningRate();
			float getMomentum();
			float getPreviousBiasChange();
			void getPreviousWeightChanges(std::list<float>&);
			float getWeightDecay();
			void getWeights(std::list<float>&);
			/*Attempts to remove a connection to the given index. Returns false if one isn't found*/
			bool removeConnection(int);
			void setBias(float);
			void setDropRatePercent(float);
			void setLearningRate(float);
			void setMomentum(float);
			void setPreviousBiasChange(float);
			void setPreviousWeightChanges(std::list<float>&);
			void setWeightDecay(float);
			void setWeights(std::list<float>&);

		private:
			//The bundle of the activation function used by this neuron.
			activationFunctionInfo actFunc;
			//The bias value the neuron uses when calculating its value.
			float bias;
			//The weight of each connection this neuron is connected to.
			std::list<float> connectionWeights;
			//What percentage of the time the value from the neuron is set to 0 regardless of input.
			float dropRatePercent;
			float learningRate;
			float momentum;
			//The amount the bias changed last time it was changed.
			float previousBiasChange;
			//The amount each weight was changed last time they were all changed.
			std::list<float> previousWeightChange;
			//The raw value of the neuron.
			std::vector<float> rawValues;
			float weightDecay;
		};

	private:
	};

}
#endif
