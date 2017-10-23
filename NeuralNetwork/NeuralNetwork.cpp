//
// Created by Asher Gunsay on 10/4/17.
//

/**
* Enter the class description here
*/


#include "NeuralNetwork.hpp"



NeuralNetwork::NeuralNetwork(std::vector<NNLayer> layers){
  layerInfo = layers;

  std::normal_distribution<double> d(0,0.3333333);
  distribution = d;

  // Initialize the weights to something. Guassian distribution with mean of 0, -1 and 1 at 3 std dev marks best
  // fine tuning, but will be more likely to change.
  // Also initialize the bias to start with initial weights of 1.
  auto initialSize = layers.size() -1;
  weights = new Matrix*[initialSize];
  bias = new Matrix*[initialSize];
  // The first layer is inputs! It will not have weights
  for(int i = 1; i < layers.size(); i++){
    weights[i-1] = new Matrix(layers[i].neuronCount, layers[i-1].neuronCount);
    bias[i-1] = new Matrix(layers[i].neuronCount, 1);
    for(unsigned int j = 0; j < weights[i-1]->rowCount(); j++){
      bias[i-1]->set(j, 0, 0.1);
      for(unsigned int k = 0; k < weights[i-1]->colCount(); k++){
        weights[i-1]->set(j, k, gaussianRandom());
      }
    }

  }



}
NeuralNetwork::~NeuralNetwork(){
  // Clean up pointers
  for(unsigned int i = 0; i < layerInfo.size() - 1; i++){
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

void NeuralNetwork::train(std::vector<Matrix> &inputs, std::vector<Matrix> &targets){
  auto size = layerInfo.size() - 1;
  std::vector< std::vector<Matrix> > outputs = {};
  std::vector<Matrix>  derivatives = {};


  std::vector< std::vector<Matrix> > errors = {};

  // For each training sample

  for(unsigned int sample = 0; sample < inputs.size(); sample++){

    // 1. Feed forward


    auto calculatedOutput = feedForward(inputs[sample], outputs[sample]);

    // 2. Calculate the Activation Function's derivative


    for(unsigned int layer = 0; layer < size; layer++){

      derivatives[layer] = setActivationDerivatives(outputs[sample][layer], layer);

    }

    // 3. Calculate the errors



    setErrors(calculatedOutput, targets[sample], errors[sample], derivatives);
  }

  // Now that all errors are calculated for the set, update the weights.
  updateWeights(errors, outputs, inputs);

}

Matrix NeuralNetwork::processImage(Matrix &input){
  Matrix blah;
  return blah;
}


Matrix NeuralNetwork::feedForward(Matrix &input, std::vector<Matrix> &activatedOutputs){
  Matrix blah;
  return blah;
}
void NeuralNetwork::setWeightedInput(Matrix &input, std::vector<Matrix> &output, unsigned int layerNumber){

}
void NeuralNetwork::setActivations(Matrix &input, std::vector<Matrix> &output, unsigned int layerNumber){

}
Matrix NeuralNetwork::setActivationDerivatives(Matrix input, unsigned int layerNumber){
  Matrix blah;
  return blah;
}
void NeuralNetwork::setErrors(Matrix &outputs, Matrix &target, std::vector<Matrix> &errors, std::vector<Matrix> &derivatives){

}
void NeuralNetwork::updateWeights(std::vector< std::vector<Matrix> > &errors, std::vector< std::vector<Matrix> > &outputs, std::vector<Matrix> &input){

}


double NeuralNetwork::gaussianRandom(){
  return distribution(engine);
}
