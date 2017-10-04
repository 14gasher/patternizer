//
// Created by Asher Gunsay on 9/22/17.
//

#include "SimpleNeuralNetwork.hpp"
#include <vector>
#include <math.h>
#include <iostream>



SimpleNeuralNetwork::SimpleNeuralNetwork(unsigned int* sizes, double learnRate = 0.1) :
  size(sizes), learningRate(learnRate)
{
  // Initialize the weights to something. Guassian distribution with mean of 0, -1 and 1 at 3 std dev marks best
  // fine tuning, but will be more likely to change.
  // Also initialize the bias to start with initial weights of 1.
  weights = new Matrix*[layers -1];
  bias = new Matrix*[layers-1];
  // The first layer is inputs! It will not have weights
  for(int i = 1; i < layers; i++){
    weights[i-1] = new Matrix(sizes[i], sizes[i-1]);
    bias[i-1] = new Matrix(sizes[i], 1);
    for(unsigned int j = 0; j < weights[i-1]->rowCount(); j++){
      bias[i-1]->set(j, 0, 0.1);
      for(unsigned int k = 0; k < weights[i-1]->colCount(); k++){
        weights[i-1]->set(j, k, fRand());
      }
    }

  }

  std::normal_distribution<double> d(0,0.3333333);
  distribution = d;

};

SimpleNeuralNetwork::~SimpleNeuralNetwork()
{
  // Clean up pointers
  for(unsigned int i = 0; i < layers - 1; i++){
    delete weights[i];
    delete bias[i];
    weights[i] = nullptr;
    bias[i] = nullptr;
  }
  delete []weights;
  delete []bias;
  weights = nullptr;
  bias = nullptr;

}


// Component Functions
double SimpleNeuralNetwork::logarithmicActivation(double weightedInput){
  return 1.0/(1.0+exp(weightedInput * (-1.0)));
};

double SimpleNeuralNetwork::logarithmicActivationDerivative(double activatedOutput){
  return activatedOutput*(1 - activatedOutput);
};


// Functions for matrix handling
void SimpleNeuralNetwork::matrixCalculateWeightedInput(Matrix &input, Matrix* output, unsigned int layerNumber) {
  Matrix withoutBias = weights[layerNumber]->matrixMultiplation(input);
  Matrix biasLayer = *(bias[layerNumber]);
  output[layerNumber] = withoutBias.addition(biasLayer);
};


void SimpleNeuralNetwork::matrixLogarithmicActivation(Matrix &weightedInput, Matrix* output, unsigned int layerNumber) {
  Matrix layerOutput(weightedInput.rowCount(), 1);

  for(unsigned int i = 0; i < weightedInput.rowCount(); i++){
    for(unsigned int j = 0; j < weightedInput.colCount(); j++){
      layerOutput.set(i,j, logarithmicActivation(weightedInput.get(i,j)));
    }
  }
  output[layerNumber] = layerOutput;

};

Matrix SimpleNeuralNetwork::matrixLogarithmicActivationDerivative(Matrix weightedInput) {
  Matrix activationDerivative(weightedInput.rowCount(), weightedInput.colCount());
  for(unsigned int i = 0; i < weightedInput.rowCount(); i++){
    for(unsigned int j = 0; j < weightedInput.colCount(); j++){
      activationDerivative.set(i,j, logarithmicActivationDerivative(weightedInput.get(i,j)));
    }
  }
  return activationDerivative;
};







// This will set the errors array order in the order that it calculates them, so output neurons first, followed by L - 1, etc
void SimpleNeuralNetwork::matrixCostFunction(Matrix &outputs, Matrix &target, Matrix* errors, Matrix* activateDerivative) {

  auto endOfMatrix = layers - 1;
  auto endTargetCol = target.scalar(-1);
  auto errorSum = outputs.addition(endTargetCol);
  auto endSigmaPrime = activateDerivative[endOfMatrix - 1];

  errors[0] = errorSum.hadamardProduct(endSigmaPrime);

  Matrix transposeOfWeights, neuralError, weightsAndError, sigmaPrimeOfWeight;
  for(int i = 1 ; i < endOfMatrix; i++){

    transposeOfWeights = weights[endOfMatrix -i]->transposition();
    neuralError = errors[i-1];
    sigmaPrimeOfWeight = activateDerivative[endOfMatrix - i - 1];

    errors[i] = transposeOfWeights.matrixMultiplation(neuralError).hadamardProduct(sigmaPrimeOfWeight);

  }

};









Matrix SimpleNeuralNetwork::feedForward(Matrix &input, Matrix* activatedOutputs){
  Matrix weightedInputs[layers-1];
  for(unsigned int layer = 0; layer < layers - 1; layer++){
    auto toPutIn = (layer == 0)? input: activatedOutputs[layer - 1];

    matrixCalculateWeightedInput(toPutIn, weightedInputs, layer);
    matrixLogarithmicActivation(weightedInputs[layer], activatedOutputs, layer);
  }
  return activatedOutputs[layers-2];
}

Matrix SimpleNeuralNetwork::processImage(Matrix &input)
{
  Matrix weightedInputs[layers-1];
  Matrix activatedOutputs[layers-1];
  for(unsigned int layer = 0; layer < layers - 1; layer++){
    Matrix toPutIn = (layer == 0)? input: activatedOutputs[layer - 1];
    matrixCalculateWeightedInput(toPutIn, weightedInputs, layer);
    matrixLogarithmicActivation(weightedInputs[layer], activatedOutputs, layer);
  }

  Matrix answer = activatedOutputs[layers-2];

  // Clean up dynamic memory
//  for(int layer = 0; layer < layers-1; layers++){
//    delete activatedOutputs[layer];
//  }
//  delete activatedOutputs;

  return answer;

}




void SimpleNeuralNetwork::updateWeights(Matrix errors[][layers -1], Matrix activatedOutputs[][layers - 1], unsigned int sampleSize, Matrix *input)
{
  // Update Weights

  for(int layer = 0; layer < layers - 1; layer++){
    Matrix layerError;
    Matrix biasError(bias[layer]->rowCount(), 1);



    for(int sample = 0; sample < sampleSize; sample++){

      Matrix transposedOutput;


      if(layer == layers - 2){
        transposedOutput = input[sample].transposition();
      } else {
        transposedOutput = activatedOutputs[sample][layers - 3 - layer].transposition();
      }

      auto sampleLayerError = errors[sample][layer].matrixMultiplation(transposedOutput);



      if(sample == 0){
        layerError = sampleLayerError;
      } else {
        layerError.addition(sampleLayerError);
      }

      biasError.addition(errors[sample][layers - 2 - layer]);

    }
    double modifiedLearningRate = learningRate/sampleSize * -1.0;


    layerError = layerError.scalar(modifiedLearningRate);
    *(weights[layers - 2 - layer]) = weights[layers - 2 - layer]->addition(layerError);

    biasError = biasError.scalar(modifiedLearningRate);
    *(bias[layer]) = bias[layer]->addition(biasError);

  }


}


void SimpleNeuralNetwork::trainWithSets(Matrix* inputs, Matrix* targets, unsigned int sampleCount) {
  Matrix activatedOutputs[sampleCount][layers - 1];
  Matrix activatedDerivative[layers-1];
  Matrix errorsForAll[sampleCount][layers-1];

  // For each training sample

  for(unsigned int sample = 0; sample < sampleCount; sample++){

    // 1. Feed forward


    auto calculatedOutput = feedForward(inputs[sample], activatedOutputs[sample]);

    // 2. Calculate the Activation Function's derivative


    for(unsigned int layer = 0; layer < layers - 1; layer++){

      activatedDerivative[layer] = matrixLogarithmicActivationDerivative(activatedOutputs[sample][layer]);

    }

    // 3. Calculate the errors



    matrixCostFunction(calculatedOutput, targets[sample], errorsForAll[sample], activatedDerivative);
  }

  // Now that all errors are calculated for the set, update the weights.
  updateWeights(errorsForAll, activatedOutputs, sampleCount, inputs);

  // Finally, clean up that memory junk...



};


double SimpleNeuralNetwork::fRand()
{
  return distribution(engine);

}