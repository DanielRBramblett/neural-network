//Copyright(C) 2020 "Daniel Bramblett" <daniel.r.bramblett@gmail.com>
#include "stdafx.h"
#include "testNeuralNetwork.h"

testNeuralNetwork::testCell::testCell(bool propFurther, int newIndex):cell(propFurther, newIndex)
{
}

void testNeuralNetwork::testCell::backwardPropagate(std::list<std::vector<float>> &x, int a, std::list<std::vector<float>> &y, std::mutex &z)
{
}

void testNeuralNetwork::testCell::copy(cell *&ptr)
{

}

void testNeuralNetwork::testCell::forwardPropagate(std::list<std::vector<float>> &x, int a)
{
}

testNeuralNetwork::testNeuron::testNeuron(bool propFurther, int newIndex):neuron(propFurther, newIndex)
{

}