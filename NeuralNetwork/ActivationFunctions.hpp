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
  static double identityAct(double x);
  static double identityDrv(double x);

  /**
   * Binary Step function
   * f(x) = { 0 for x < 0; 1 for x >= 0 }
   * f'(x) = { 0 for x != 0; nan for x == 0 }
   *
   * range {0, 1}
   * Continuous except where x == 0
   */
  static double binaryStepAct(double x);
  static double binaryStepDrv(double x);

  /**
   * Logistic or Soft Step
   * f(x) = 1 / (1 + e ^ {-x} )
   * f'(x) = f(x)(1 - f(x))
   *
   * range (0, 1)
   * Continuous
   */
  static double logarithmicAct(double x);
  static double logarithmicDrv(double x);

  /**
   * Hyperbolic Tangent
   * f(x) = (e ^ x - e ^ {-x}) / (e ^ x + e ^ {-x})
   * f'(x) = 1 - f(x)^2
   *
   * range (-1, 1)
   * Continuous
   */
  static double tanhAct(double x);
  static double tanhDrv(double x);

  /**
   * ArcTangent
   * f(x) = tan^{-1}(x)
   * f'(x) = 1 / (x^2 + 1)
   *
   * range (-pi/2, pi/2)
   * Continuous
   */
  static double arctanAct(double x);
  static double arctanDrv(double x);

  /**
   * Soft Sign
   * f(x) = x / (1 + |x|)
   * f'(x) = 1 / (1 + |x|)^2
   *
   * range (-1, 1)
   * Continuous except x = 0
   */
  static double softSignAct(double x);
  static double softSignDrv(double x);

  /**
   * Rectified Linear Unit
   * f(x) = { 0 for x < 0; x for x >= 0 }
   * f'(x) = { 0 for x < 0; 1 for x >=1 }
   *
   * range (-inf, inf)
   * Continuous except x = 0
   */
  static double reluAct(double x);
  static double reluDrv(double x);

  /**
   * Leaky Rectified Linear Unit
   * f(x) = { 0.01x for x < 0; x for x >= 0 }
   * f'(x) = { 0.01 for x < 0; 1 for x >=1 }
   *
   * range (-inf, inf)
   * Continuous except x = 0
   */
  static double lreluAct(double x);
  static double lreluDrv(double x);

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
  static double preluAct(double x, double a);
  static double preluDrv(double x, double a);

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
  static double eluAct(double x, double a, double v);
  static double eluDrv(double x, double a, double v);

  /**
   * Soft Plus
   * f(x) = ln(1 + e^x)
   * f'(x) = 1 / (1 + e^{-x})
   *
   * range (0, inf)
   * Continuous
   */
  static double softPlusAct(double x);
  static double softPlusDrv(double x);

  /**
   * Bent Identity
   * f(x) = ((x^2 + 1)^{1/2} - 1)/2 + x
   * f'(x) = ( x) / (2(x^2 + 1)^{1/2}) + 1
   *
   * range (-inf, inf)
   * Continuous
   */
  static double bentIdentityAct(double x);
  static double bentIdentityDrv(double x);

};


#endif //NEURALNETWORKS_ACTIVATIONFUNCTIONS_HPP
