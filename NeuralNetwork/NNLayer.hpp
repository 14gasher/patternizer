//
// Created by Asher Gunsay on 10/23/17.
//

/**
* Enter the class description here
*/


#ifndef NEURALNETWORKS_NNLAYER_HPP
#define NEURALNETWORKS_NNLAYER_HPP


#include <functional>

struct NNLayer
{
  std::function<double (double)> activation;
  std::function<double (double)> derivative;
  unsigned int neuronCount;

};


#endif //NEURALNETWORKS_NNLAYER_HPP
