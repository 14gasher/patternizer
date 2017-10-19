//
// Created by Asher Gunsay on 9/22/17.
//

#ifndef NEURALNETWORKS_SIMPLENEURALNETWORK_HPP
#define NEURALNETWORKS_SIMPLENEURALNETWORK_HPP

#include "../HelperClasses/Matrix.hpp"
#include <random>



class SimpleNeuralNetwork
{
public:
  /**
   * Constructor
   *
   * @param sizes Array of sizes
   * @param learnRate rate at which nn will learn. More is faster but less accurate.
   */
  SimpleNeuralNetwork(unsigned int* sizes, double learnRate, unsigned int epochCount);

  /**
   * Deconstructor
   */
  ~SimpleNeuralNetwork();

  /**
   * Trains according to a set (theoretically faster because of matrix operations... slower for me because I'm stubborn)
   * @param inputs
   * @param targets
   * @param sampleCount
   */
  void trainWithSets(Matrix* inputs, Matrix* targets, unsigned int sampleCount);

  /**
   * Processes input and outputs a result.
   *
   * @param input
   * @param activatedOutputs layer by layer output matrix array
   * @return
   */
  Matrix feedForward(Matrix &input, Matrix* activatedOutputs);

  /**
   * Processes input and outputs a classification matrix.
   * @param input
   * @return
   */
  Matrix processImage(Matrix &input);


  /**
   * Gets a visual representation of the first layer of weights (because it's cool) and saves it to a ppm
   * upon destruction of the network
   */
   void addCurrentWeightsToFile();


private:
  unsigned int* size;
  static const unsigned int layers = 3;
  Matrix** weights;
  Matrix** bias;
  const double learningRate;




  // Functions for doing things 1 input at a time

  /**
   * We are using the logarithmic function here, or 1 / (1 + e ^ -x )
   *
   * @param weightedInput
   * @return
   */
  double logarithmicActivation(double weightedInput);

  /**
   * Derivative of the logarithmic function, or f(x)(1 - f(x))
   * @param weightedInput
   * @return
   */
  double logarithmicActivationDerivative(double weightedInput);



  // Functions for matrix handling

  /**
   * Calculates the weighted inputs for each layer
   *
   * @param input
   * @param output
   * @param layerNumber
   */
  void matrixCalculateWeightedInput(Matrix &input, Matrix* output, unsigned int layerNumber);

  /**
   * Applies the logarithmic activation on a matrix level
   *
   * @param weightedInput
   * @param output
   * @param layerNumber
   */
  void matrixLogarithmicActivation(Matrix &weightedInput, Matrix* output, unsigned int layerNumber);

  /**
   * Applies the logarithmic derivative on a matrix level
   *
   * @param weightedInput
   * @return
   */
  Matrix matrixLogarithmicActivationDerivative(Matrix weightedInput);

  /**
   * Calculates errors in each layer
   *
   * @param outputs
   * @param target
   * @param errors
   * @param activateDerivative
   */
  void matrixCostFunction(Matrix &outputs, Matrix &target, Matrix* errors, Matrix* activateDerivative);


  /**
   * Changes weights and biases in each layer
   *
   * @param activatedOutputs
   * @param sampleSize
   * @param input
   */
  void updateWeights(Matrix[][layers - 1], Matrix activatedOutputs[][layers -1 ], unsigned int sampleSize, Matrix *input);

  // Number Generator
  double fRand();
  std::default_random_engine engine;
  std::normal_distribution<double> distribution;

  std::string weightsString = "";
  std::string weightsString2 = "";






};


#endif //NEURALNETWORKS_SIMPLENEURALNETWORK_HPP
