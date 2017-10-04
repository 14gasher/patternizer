//
// Created by Asher Gunsay on 9/23/17.
//

#ifndef NEURALNETWORKS_MINSTHELPER_HPP
#define NEURALNETWORKS_MINSTHELPER_HPP


#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#include "Matrix.hpp"

class MINSTHelper{
public:
  void ReadMNIST(int NumberOfImages, std::string filepath);
  void ReadMNISTLabels(int NumberOfImages, std::string filepath);

  Matrix* getInputAtIndex(unsigned int index){return inputs[index];};
  Matrix* getTargetAtIndex(unsigned int index){return targets[index];};


private:
  Matrix** inputs;
  Matrix** targets;


  unsigned int ReverseInt (unsigned int i);


};



#endif //NEURALNETWORKS_MINSTHELPER_HPP
