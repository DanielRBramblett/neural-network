//Copyright(C) 2020 "Daniel Bramblett" <daniel.r.bramblett@gmail.com>
/*Header containing the prototype for the neuralNetwork class along including the prototypes of
  all the nested classes inside of it.*/
#ifndef NEURAL_NET_LIB
#define NEURAL_NET_LIB


#include<list>
#include<mutex>
#include<vector>
#include "activationFunctions.h"
#include "preprocessorFlags.h"


//Default values used.
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

protected:
	/*Nested abstract cell class which represents each cell in the neural network.*/
	class cell
	{
	public:
		cell(bool, int);
		//Callable functions:
		virtual bool addConnection(int);
		bool canUpdate(const std::vector<bool>&);
		void getConnections(std::list<int>&);
		int getIndex();
		bool getPropagateFurther();
		bool removeConnection(int);
		void setPropagateFurther(bool);
		
		//Pure virutal functions:
		virtual void backwardPropagate(std::list < std::vector<float>>&, int, std::list<std::vector<float>>&, std::mutex&) = 0;
		virtual void forwardPropagate(std::list < std::vector<float>>&, int) = 0;
	protected:
		//Whether the error needs to be back propagated further.
		bool backPropagateFurther;
		int cellIndex;
		//List of cells connections.
		std::list<int> connections;
	};

	class neuron : public cell
	{
	public:
		neuron(bool, int);
		bool addConnection(int);
		bool addConnection(int, float);
		void backwardPropagate(std::list < std::vector<float>>&, int, std::list<std::vector<float>>&, std::mutex&);
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
		activationFunctionInfo actFunc;
		float bias;
		std::list<float> connectionWeights;
		float dropRatePercent;
		float learningRate;
		float momentum;
		float previousBiasChange;
		std::list<float> previousWeightChange;
		std::vector<float> rawValues;
		float weightDecay;
	};

private:
};


#endif
