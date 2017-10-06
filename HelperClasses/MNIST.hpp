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


/**
 *
 * This class is responsible for the following processes:
 *
 * 1. Reading in MINST data
 * 2. Convert data from big to little endian
 * 3. Store data into a Matrix array
 * 4. Reading in MINST labels
 * 5. Convert labels from big to little endian
 * 6. Store labels into a Matrix array
 * 7. Data and label retrieval
 *
 */

class MNIST{
public:
  /**
   * Reads in MNIST data and stores it
   *
   * @param NumberOfImages
   * @param filepath
   */
  void ReadMNIST(unsigned int NumberOfImages, std::string filepath);

  /**
   * Reads in MINST labels and stores it
   *
   * @param NumberOfImages
   * @param filepath
   */
  void ReadMNISTLabels(unsigned int NumberOfImages, std::string filepath);

  /**
   * Gets an image in Matrix form from MINST data
   *
   * @param index
   * @return image sample
   */
  Matrix* getInputAtIndex(unsigned int index){return inputs[index];};

  /**
   * Gets an image's label in Matrix form
   *
   * @param index
   * @return image label
   */
  Matrix* getTargetAtIndex(unsigned int index){return targets[index];};

  ~MNIST(){
    for(int i = 0; i < numberOfImages; i++){
      delete inputs[i];
      delete targets[i];
    }
    delete []inputs;
    delete []targets;
  }


private:
  /**
   * Holds the images as a Matrix array
   */
  Matrix** inputs;
  /**
   * Holds the labels for the images as Matrices in an array
   */
  Matrix** targets;

  unsigned int numberOfImages = 0;


  /**
   * Helper function responsible for converting endians
   * @param i
   * @return
   */
  unsigned int ReverseInt (unsigned int i);


};



#endif //NEURALNETWORKS_MINSTHELPER_HPP
