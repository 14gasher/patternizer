//
// Created by Asher Gunsay on 10/4/17.
//

/**
* A better implementation of the Neural network.
 *
 * Key upgrades include:
 *
 * Passing in an input matrix as opposed to an input array
 * The ability to choose your cost function
 * The ability to regularize
 * The ability to add momentum
 *
*/


#ifndef NEURALNETWORKS_NEURALNETWORK_HPP
#define NEURALNETWORKS_NEURALNETWORK_HPP

#include <vector>
#include <random>

#include "NNLayer.hpp"
#include "../HelperClasses/Matrix.hpp"



class NeuralNetwork
{
public:
  NeuralNetwork(std::vector<NNLayer> layers);
  ~SimpleNeuralNetwork();

  void train(Matrix* inputs, Matrix* targets, unsigned int sampleCount);

  Matrix processImage(Matrix &input);

private:

  Matrix feedForward(Matrix &input, Matrix* activatedOutputs);
  void setWeightedInput(Matrix &input, Matrix* output, unsigned int layerNumber);
  void setActivations(Matrix &weightedInput, Matrix* output, unsigned int layerNumber);
  Matrix setActivationDerivatives(Matrix weightedInput);
  void setErrors(Matrix &outputs, Matrix &target, Matrix* errors, Matrix* activateDerivative);
  void updateErrors(Matrix** errors, Matrix** activatedOutputs, unsigned int sampleSize, Matrix *input);


  double gaussianRandom();
  std::default_random_engine engine;
  std::normal_distribution<double> distribution;


  std::vector<NNLayer> layerInfo;


};


#endif //NEURALNETWORKS_NEURALNETWORK_HPP
