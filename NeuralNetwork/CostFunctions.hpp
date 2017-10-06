//
// Created by Asher Gunsay on 10/4/17.
//

/**
* I am considering implementing this to enable better code reuse... It may be moved from NeuralNetwork over to HelperClasses
 *
 *
*/


#ifndef NEURALNETWORKS_COSTFUNCTIONS_HPP
#define NEURALNETWORKS_COSTFUNCTIONS_HPP

#include "../HelperClasses/Matrix.hpp"

/**
 * Each of these need to be hadamarand product-ed with the derivative of the activation function to calculate the
 * error in the output layer
 */
class CostFunctions
{
public:
  /**
   * Gradient in terms of output of the activation function:
   * (a - y)
   * @return
   */
  static Matrix quadraticCost();


  /**
   * Gradient in terms of output of the activation function:
   * (a - y) / ( (1-a)*(a) )
   * @return
   */
  static Matrix crossEntropyCost();


  /**
   * Gradient in terms of output of the activation function:
   * (2 / t) (a - y) * exponentialCostScalar
   *
   * t is some arbitrary value. Try different things to make it work better
   * @return
   */
  static Matrix exponentialCost();


  /**
   * Gradient in terms of output of the activation function:
   * ( (a)^{1/2} - (y)^{1/2} ) / ( (2a)^{1/2} )
   * @return
   */
  static Matrix hellingerCost();


  /**
   * Gradient in terms of output of the activation function:
   * ( -y / a )
   * @return
   */
  static Matrix kullbackLeiberCost();


  /**
   * Gradient in terms of output of the activation function:
   * ( -y + a ) / y
   * @return
   */
  static Matrix generalizedKullbackLeiberCost();


  /**
   * Gradient in terms of output of the activation function:
   * ( y - a ) / a^2
   *
   * where a^2 is squaring the components of a...
   * @return
   */
  static Matrix itakuraSaitoCost();

private:

  // Constructor private to avoid instantiation of the class

  /**
   * t( exp ( 1/t (sum( ( i_{j} - e_{j} ) ^ 2 ) ) )
   *
   * @return
   */
  double exponentialCostScalar();

  CostFunctions(){};

};


#endif //NEURALNETWORKS_COSTFUNCTIONS_HPP
