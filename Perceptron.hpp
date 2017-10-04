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
  Perceptron(unsigned int inputCount);
  ~Perceptron();
  int processInput(float* inputs);
  bool train(float* trainInputs, int desired);
};

class Trainer{
private:
  float* inputs = nullptr;
  float slope;
  float yIntercept;
public:
  Trainer(Perceptron& p, unsigned int inputCount, unsigned int desiredTrainingPoints, float m, float b);
  int linearAnswer(float x, float z);
  ~Trainer();
};


#endif //NEURALNETWORKS_PERCEPTRON_HPP
