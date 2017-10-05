//
// Created by Asher Gunsay on 10/4/17.
//

/**
* Options to use for activations. May be moved from current home over to HelperClasses or to a FunctionsFolder
*/


#ifndef NEURALNETWORKS_ACTIVATIONFUNCTIONS_HPP
#define NEURALNETWORKS_ACTIVATIONFUNCTIONS_HPP


#include "../HelperClasses/Matrix.hpp"

class ActivationFunctions
{

public:


private:
  /**
   * Constructor is private to avoid instantiation of this static class
   */
  ActivationFunctions(){};


  /**
   * Identity function
   * f(x) = x
   * f'(x) = 1
   *
   * range (-inf, inf)
   * Continuous
   */
  static double identityAct();
  static double identityDrv();

  /**
   * Binary Step function
   * f(x) = { 0 for x < 0; 1 for x >= 0 }
   * f'(x) = { 0 for x != 0; nan for x == 0 }
   *
   * range {0, 1}
   * Continuous except where x == 0
   */
  static double binaryStepAct();
  static double binaryStepDrv();

  /**
   * Logistic or Soft Step
   * f(x) = 1 / (1 + e ^ {-x} )
   * f'(x) = f(x)(1 - f(x))
   *
   * range (0, 1)
   * Continuous
   */
  static double logarithmicAct();
  static double logarithmicDrv();

  /**
   * Hyperbolic Tangent
   * f(x) = (e ^ x - e ^ {-x}) / (e ^ x + e ^ {-x})
   * f'(x) = 1 - f(x)^2
   *
   * range (-1, 1)
   * Continuous
   */
  static double tanhAct();
  static double tanhDrv();

  /**
   * ArcTangent
   * f(x) = tan^{-1}(x)
   * f'(x) = 1 / (x^2 + 1)
   *
   * range (-pi/2, pi/2)
   * Continuous
   */
  static double arctanAct();
  static double arctanDrv();

  /**
   * Soft Sign
   * f(x) = x / (1 + |x|)
   * f'(x) = 1 / (1 + |x|)^2
   *
   * range (-1, 1)
   * Continuous except x = 0
   */
  static double softSignAct();
  static double softSignDrv();

  /**
   * Rectified Linear Unit
   * f(x) = { 0 for x < 0; x for x >= 0 }
   * f'(x) = { 0 for x < 0; 1 for x >=1 }
   *
   * range (-inf, inf)
   * Continuous except x = 0
   */
  static double reluAct();
  static double reluDrv();

  /**
   * Leaky Rectified Linear Unit
   * f(x) = { 0.01x for x < 0; x for x >= 0 }
   * f'(x) = { 0.01 for x < 0; 1 for x >=1 }
   *
   * range (-inf, inf)
   * Continuous except x = 0
   */
  static double lreluAct();
  static double lreluDrv();

  /**
   * Parametric Rectified Linear Unit
   * f(x) = { ax for x < 0; x for x >= 0 }
   * f'(x) = { a for x < 0; 1 for x >=1 }
   *
   * range (-inf, inf)
   * Continuous except x = 0
   *
   * a becomes a learned value as well...
   */
  static double preluAct();
  static double preluDrv();

  /**
   * Exponential Linear Unit
   * f(a, x) = v { a (e^x - 1) for x < 0; x for x >= 0 }
   * f'(a, x) = v { f(a,x) + a for x < 0; 1 for x >= 0 }
   *
   * range (-a, inf)
   * Continuous except x = 0
   *
   * a can become a hyper parameter, or make this
   * a Scaled exponential Linear Unit by setting
   * v = 1.0507 and a = 1.67326, weights are initialized in a special way...
   * Look it up again...
   */
  static double eluAct();
  static double eluDrv();

  /**
   * Soft Plus
   * f(x) = ln(1 + e^x)
   * f'(x) = 1 / (1 + e^{-x})
   *
   * range (0, inf)
   * Continuous
   */
  static double softPlusAct();
  static double softPlusDrv();

  /**
   * Bent Identity
   * f(x) = ((x^2 + 1)^{1/2} - 1)/2 + x
   * f'(x) = ( x) / (2(x^2 + 1)^{1/2}) + 1
   *
   * range (-inf, inf)
   * Continuous
   */
  static double bentIdentityAct();
  static double bentIdentityDrv();

};


#endif //NEURALNETWORKS_ACTIVATIONFUNCTIONS_HPP
