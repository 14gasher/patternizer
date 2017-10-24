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
  ~NeuralNetwork();

  void train(std::vector<Matrix> &inputs, std::vector<Matrix> &targets);

  Matrix processImage(Matrix &input);

private:

  Matrix feedForward(Matrix &input, std::vector<Matrix> &activatedOutputs, std::vector<Matrix> &weightedInputs);
  Matrix setWeightedInput(Matrix &input, unsigned int layerNumber);
  Matrix setActivations(Matrix &weightedInput, unsigned int layerNumber);
  Matrix setActivationDerivatives(Matrix weightedInput, unsigned int layerNumber);
  std::vector<Matrix> setErrors(Matrix &outputs, Matrix &target, std::vector<Matrix> &activateDerivative);
  void updateWeights(std::vector< std::vector<Matrix> > &errors, std::vector< std::vector<Matrix> > &activatedOutputs, std::vector<Matrix> &input);


  double gaussianRandom();
  std::default_random_engine engine;
  std::normal_distribution<double> distribution;


  std::vector<NNLayer> layerInfo;
  Matrix** weights;
  Matrix** bias;


};


#endif //NEURALNETWORKS_NEURALNETWORK_HPP
