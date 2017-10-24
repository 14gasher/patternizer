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
      bias[i-1]->set(j, 0, gaussianRandom());
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




  std::vector< std::vector<Matrix> > errors = {};

  // For each training sample

  for(unsigned int sample = 0; sample < inputs.size(); sample++){


    // 1. Feed forward
    std::vector<Matrix> blankM = {};
    outputs.push_back(blankM);

    std::vector<Matrix> weightedInputs = {};
    auto calculatedOutput = feedForward(inputs[sample], outputs[sample], weightedInputs);

    // 2. Calculate the Activation Function's derivative

    std::vector<Matrix>  derivatives = {};
    for(unsigned int layer = 0; layer < size; layer++){
      auto derv = setActivationDerivatives(weightedInputs[layer], layer);
      derivatives.push_back(derv);
    }




    // 3. Calculate the errors
    auto err = setErrors(calculatedOutput, targets[sample], derivatives);

    errors.push_back( err );
  }

  // Now that all errors are calculated for the set, update the weights.
  updateWeights(errors, outputs, inputs);

}

Matrix NeuralNetwork::processImage(Matrix &input){
  std::vector<Matrix> filler = {};
  std::vector<Matrix> filler2 = {};
  return feedForward(input, filler, filler2);
}


Matrix NeuralNetwork::feedForward(Matrix &input, std::vector<Matrix> &activatedOutputs, std::vector<Matrix> &weightedInputs){
  auto bound = layerInfo.size() - 1;
  auto toUse = input;
  for(unsigned int layer = 0; layer < bound; layer++){
    Matrix weightedInput( setWeightedInput(toUse, layer) );
    toUse = setActivations(weightedInput, layer);
    activatedOutputs.push_back(toUse);
    weightedInputs.push_back(weightedInput);
  }
  return toUse;
}


Matrix NeuralNetwork::setWeightedInput(Matrix &input, unsigned int layerNumber){
  Matrix beforeBias = weights[layerNumber]->matrixMultiplation(input);
  Matrix biasLayer = *(bias[layerNumber]);
  Matrix afterBias = beforeBias.addition(biasLayer);
  return afterBias;
}
Matrix NeuralNetwork::setActivations(Matrix &input, unsigned int layerNumber){
  Matrix output(input.rowCount(), 1);
  for(unsigned int row = 0; row < input.rowCount(); row++){
    for(unsigned int col = 0; col < input.colCount(); col++){
      auto c = input.get(row, col);
      auto res = layerInfo[layerNumber].activation(c);
      output.set(row, col, res);
    }
  }
  return output;
}

Matrix NeuralNetwork::setActivationDerivatives(Matrix input, unsigned int layerNumber){
  Matrix output(input.rowCount(), 1);
  for(unsigned int row = 0; row < input.rowCount(); row++){
    for(unsigned int col = 0; col < input.colCount(); col++){
      auto c = input.get(row, col);
      auto res = layerInfo[layerNumber].derivative(c);
      output.set(row, col, res);
    }
  }
  return output;
}
std::vector<Matrix> NeuralNetwork::setErrors(Matrix &outputs, Matrix &target, std::vector<Matrix> &derivatives){
  std::vector<Matrix> e = {};
  unsigned int bound = layerInfo.size() - 1;
  Matrix endGoal = target.scalar(-1);
  Matrix errorSum = outputs.addition(endGoal);
  Matrix endDerivative = derivatives[bound - 1];

  e.push_back(errorSum.hadamardProduct(endDerivative));

  for(unsigned int i = 1; i < bound; i++){
    Matrix transposeOfWeights = weights[bound - i]->transposition();
    Matrix layerError = e[i - 1];
    Matrix derivative = derivatives[bound - i - 1];
    e.push_back(transposeOfWeights.matrixMultiplation(layerError).hadamardProduct(derivative));
  }

  return e;

}
void NeuralNetwork::updateWeights(std::vector< std::vector<Matrix> > &errors, std::vector< std::vector<Matrix> > &outputs, std::vector<Matrix> &input){

  double learningRate = 3.0;

  auto bound = layerInfo.size() - 1;
  for(int layer = 0; layer < bound; layer++){
    Matrix layerError;
    Matrix biasError(bias[layer]->rowCount(), 1);

    for(int sample = 0; sample < input.size(); sample++){
      Matrix transposedOut;

      if(layer == bound - 1){
        transposedOut = input[sample].transposition();
      } else {
        transposedOut = outputs[sample][bound - 2 - layer].transposition();
      }

      Matrix sampleLayerError = errors[sample][layer].matrixMultiplation(transposedOut);
      if(sample == 0){
        layerError = sampleLayerError;
      } else{
        layerError = layerError.addition(sampleLayerError);
      }

      biasError.addition(errors[sample][bound-1-layer]);

    }
    double modLearningRate = learningRate / input.size() * -1.0;
    layerError = layerError.scalar(modLearningRate);
    *(weights[bound - 1 - layer]) = weights[bound - 1 -layer]->addition(layerError);

    biasError = biasError.scalar(modLearningRate);
    *(bias[layer]) = bias[layer]->addition(biasError);
  }



}


double NeuralNetwork::gaussianRandom(){
  return distribution(engine);
}
