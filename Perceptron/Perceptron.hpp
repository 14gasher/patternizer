//
// Created by Asher Gunsay on 9/18/17.
//

#ifndef NEURALNETWORKS_PERCEPTRON_HPP
#define NEURALNETWORKS_PERCEPTRON_HPP
#include <time.h>
#include <cstdlib>
#include <iostream>


class Perceptron{

private:
  float* weights = nullptr;
  float c = 0.00001;
  unsigned int inputAndWeightCount;
public:
  /**
   * Constructor
   *
   * @param inputCount
   */
  Perceptron(unsigned int inputCount);


  /**
   * Deconstructor
   */
  ~Perceptron();


  /**
   * Returns true or false based on weights
   *
   * @param inputs
   * @return
   */
  int processInput(float* inputs);

  /**
   * Trains the perceptron. Compares expected vs output and changes weight according to bias
   *
   * @param trainInputs
   * @param desired
   * @return
   */
  bool train(float* trainInputs, int desired);
};

class Trainer{
private:
  float* inputs = nullptr;
  float slope;
  float yIntercept;
public:
  /**
   * Constructor. This is where the training will take place.
   *
   * @param p
   * @param inputCount
   * @param desiredTrainingPoints
   * @param m
   * @param b
   */
  Trainer(Perceptron& p, unsigned int inputCount, unsigned int desiredTrainingPoints, float m, float b);

  /**
   * Gets the correct answer of the given line
   * @param x
   * @param z
   * @return
   */
  int linearAnswer(float x, float z);

  /**
   * Deconstructor
   */
  ~Trainer();
};


#endif //NEURALNETWORKS_PERCEPTRON_HPP
