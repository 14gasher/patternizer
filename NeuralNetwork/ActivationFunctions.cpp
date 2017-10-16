//
// Created by Asher Gunsay on 10/4/17.
//

/**
* Implementation of the Activation function
*/

#include <math.h>

#include "ActivationFunctions.hpp"

/**
 * Private implementations
 */


/**
 * Identity function
 * f(x) = x
 * f'(x) = 1
 *
 * range (-inf, inf)
 * Continuous
 */
static double ActivationFunctions::identityAct(double x){
  return x;
}
static double ActivationFunctions::identityDrv(double x){
  return 1;
}

/**
 * Binary Step function
 * f(x) = { 0 for x < 0; 1 for x >= 0 }
 * f'(x) = { 0 for x != 0; nan for x == 0 }
 *
 * range {0, 1}
 * Continuous except where x == 0
 */
static double ActivationFunctions::binaryStepAct(double x){
  if(x < 0){
    return 0;
  } else {
    return 1;
  }
}
static double ActivationFunctions::binaryStepDrv(double x){
  // We will pretend it is continuous for now....
  return 0;
}

/**
 * Logistic or Soft Step
 * f(x) = 1 / (1 + e ^ {-x} )
 * f'(x) = f(x)(1 - f(x))
 *
 * range (0, 1)
 * Continuous
 */
static double ActivationFunctions::logarithmicAct(double x){
  return 1.0/(1.0+exp(x * (-1.0)));
}
static double logarithmicDrv(double x){
  auto result = ActivationFunctions::logarithmicAct(x);
  return result * (1 - result);
}

/**
 * Hyperbolic Tangent
 * f(x) = (e ^ x - e ^ {-x}) / (e ^ x + e ^ {-x})
 * f'(x) = 1 - f(x)^2
 *
 * range (-1, 1)
 * Continuous
 */
static double ActivationFunctions::tanhAct(double x){
  return (exp(x) - exp(-1 * x)) / (exp(x) + exp(-1 * x));
}
static double ActivationFunctions::tanhDrv(double x){
  auto result = ActivationFunctions::tanhAct(x);
  return 1 - pow(result, 2);
}

/**
 * ArcTangent
 * f(x) = tan^{-1}(x)
 * f'(x) = 1 / (x^2 + 1)
 *
 * range (-pi/2, pi/2)
 * Continuous
 */
static double ActivationFunctions::arctanAct(double x){
  return atan(x);
}
static double ActivationFunctions::arctanDrv(double x){
  return (1.0) / (pow(x, 2.0) + 1);
}

/**
 * Soft Sign
 * f(x) = x / (1 + |x|)
 * f'(x) = 1 / (1 + |x|)^2
 *
 * range (-1, 1)
 * Continuous except x = 0
 */
static double ActivationFunctions::softSignAct(double x){
  return (x) / (1 + abs(x));
}
static double ActivationFunctions::softSignDrv(double x){
  return (1.0) / pow(1 + abs(x), 2.0);
}

/**
 * Rectified Linear Unit
 * f(x) = { 0 for x < 0; x for x >= 0 }
 * f'(x) = { 0 for x < 0; 1 for x > 0, nan for x == 0 }
 *
 * range (-inf, inf)
 * Continuous except x = 0
 */
static double ActivationFunctions::reluAct(double x){
  if(x < 0){
    return 0;
  } else {
    return x;
  }
}
static double ActivationFunctions::reluDrv(double x){
  // Pretend it is continuous for now...
  if(x < 0) {
    return 0;
  } else {
    return 1;
  }
}

/**
 * Leaky Rectified Linear Unit
 * f(x) = { 0.01x for x < 0; x for x >= 0 }
 * f'(x) = { 0.01 for x < 0; 1 for x > 0, nan for x == 0 }
 *
 * range (-inf, inf)
 * Continuous except x = 0
 */
static double ActivationFunctions::lreluAct(double x){
  if(x < 0){
    return 0.01 * x;
  } else {
    return x;
  }
}
static double ActivationFunctions::lreluDrv(double x){
  // Pretend continuous for now
  if(x < 0){
    return 0.01;
  } else {
    return 1;
  }
}

/**
 * Parametric Rectified Linear Unit
 * f(x) = { ax for x < 0; x for x >= 0 }
 * f'(x) = { a for x < 0; 1 for x > 0, nan for x == 0 }
 *
 * range (-inf, inf)
 * Continuous except x = 0
 *
 * a becomes a learned value as well...
 */
static double ActivationFunctions::preluAct(double x, double a){
  if(x < 0){
    return a * x;
  } else {
    return x;
  }
}
static double ActivationFunctions::preluDrv(double x, double a){
  if(x < 0){
    return a;
  } else {
    return 1;
  }
}

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
static double ActivationFunctions::eluAct(double x, double a, double v) {

  double halfOfReturn;

  if(x < 0){
    halfOfReturn = a * (exp(x) - 1);
  } else {
    halfOfReturn = x;
  }

  return v * halfOfReturn;
}
static double ActivationFunctions::eluDrv(double x, double a, double v){
  double halfOfReturn;

  if(x < 0){
    halfOfReturn = a + ActivationFunctions::eluAct(x,a,v);
  } else {
    halfOfReturn = 1;
  }

  return v * halfOfReturn;
}

/**
 * Soft Plus
 * f(x) = ln(1 + e^x)
 * f'(x) = 1 / (1 + e^{-x})
 *
 * range (0, inf)
 * Continuous
 */
static double ActivationFunctions::softPlusAct(double x){
  return log(1 + exp(x));
}
static double ActivationFunctions::softPlusDrv(double x){
  return 1.0 / (1 + exp(-1.0 * x));
}

/**
 * Bent Identity
 * f(x) = ((x^2 + 1)^{1/2} - 1)/2 + x
 * f'(x) = ( x) / (2(x^2 + 1)^{1/2}) + 1
 *
 * range (-inf, inf)
 * Continuous
 */
static double ActivationFunctions::bentIdentityAct(double x){
  return ((pow(pow(x,2) + 1, 0.5) - 1) / 2.0) + x;
}
static double ActivationFunctions::bentIdentityDrv(double x){
  return (x) / (2 * (pow(pow(x,2) + 1, 0.5)) + 1);
}