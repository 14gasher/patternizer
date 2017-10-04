//
// Created by Asher Gunsay on 9/22/17.
//

#ifndef NEURALNETWORKS_SIMPLENEURALNETWORK_HPP
#define NEURALNETWORKS_SIMPLENEURALNETWORK_HPP

#include "Matrix.hpp"
#include <random>



class SimpleNeuralNetwork
{
public:
  // Constructor
  SimpleNeuralNetwork(unsigned int* sizes, double learnRate);
  ~SimpleNeuralNetwork();

  // Trains the network by using groups of inputs
  void trainWithSets(Matrix* inputs, Matrix* targets, unsigned int sampleCount);
  // Processes input. The weightedInputs will change an array passed to it aid in backpropagation.
  // The activatedOutputs gives the actual outputs from each level of the network.
  Matrix feedForward(Matrix &input, Matrix* activatedOutputs);
  // This is the function we will use outside the class to test things...
  Matrix processImage(Matrix &input);


private:
  unsigned int* size;
  static const unsigned int layers = 3;
  Matrix** weights;
  Matrix** bias;
  const double learningRate;




  // Functions for doing things 1 input at a time

  double logarithmicActivation(double weightedInput);
  double logarithmicActivationDerivative(double weightedInput);



  // Functions for matrix handling
  void matrixCalculateWeightedInput(Matrix &input, Matrix* output, unsigned int layerNumber);
  void matrixLogarithmicActivation(Matrix &weightedInput, Matrix* output, unsigned int layerNumber);
  Matrix matrixLogarithmicActivationDerivative(Matrix weightedInput);
  void matrixCostFunction(Matrix &outputs, Matrix &target, Matrix* errors, Matrix* activateDerivative);


  void updateWeights(Matrix[][layers - 1], Matrix activatedOutputs[][layers -1 ], unsigned int sampleSize, Matrix *input);

  // Number Generator
  double fRand();
  std::default_random_engine engine;
  std::normal_distribution<double> distribution;




};


#endif //NEURALNETWORKS_SIMPLENEURALNETWORK_HPP
