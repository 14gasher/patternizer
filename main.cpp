#include "Perceptron/Perceptron.hpp"
#include "HelperClasses/Matrix.hpp"
#include <math.h>
#include "SimpleNeuralNetwork/SimpleNeuralNetwork.hpp"
#include "HelperClasses/MNIST.hpp"
#include <iomanip>


#include "NeuralNetwork/NeuralNetwork.hpp"
#include "NeuralNetwork/ActivationFunctions.hpp"


const unsigned int POPULATION = 60000;
const unsigned int PIXEL_COUNT = 784;

/**
 * This will demonstrate the power of the perceptron.
 *
 * It will figure out the line given to it by the user,
 * then when given a point, will output whether a given point is above
 * or below a line
 */
void perceptronDemo();




/**
 * This will test the accuracy of the SimpleNeuralNetwork. It compares the given results against
 * the targets using a set of data that hasn't been backpropagated by the network.
 *
 * It then prints a report to the console.
 * The report includes overall errors and accuracy.
 * Also, broken down by digits, the expected number of each digit, guessed number, number guessed correctly, and
 * ratio of number guessed correctly to number guessed.
 *
 * @param digitProcessor
 * @param helper
 */
void nnTester(SimpleNeuralNetwork &digitProcessor, MNIST &helper);



/**
 * This will train the network through an epoch. 10,000 random images will be pulled from the helper,
 * and will be fed forward, with the error backpropagated.
 *
 * @param nn
 * @param helper
 */
void nnTrainer(SimpleNeuralNetwork &nn, MNIST &helper);




/**
 * This will train and test a neural network for a given number of epochs.
 *
 * @param epochs
 */
void nnDemo(unsigned int epochs);




















int main()
{
//  srand(time(NULL));
//
//  nnDemo(30);


  MNIST helper;

  helper.ReadMNIST(POPULATION, "train-images-idx3-ubyte");
  helper.ReadMNISTLabels(POPULATION, "train-labels-idx1-ubyte");


  NNLayer layer0;
  layer0.activation = ActivationFunctions::logarithmicAct;
  layer0.derivative = ActivationFunctions::logarithmicDrv;
  layer0.neuronCount = 28 * 28;

  NNLayer layer1;
  layer1.activation = ActivationFunctions::logarithmicAct;
  layer1.derivative = ActivationFunctions::logarithmicDrv;
  layer1.neuronCount = 15;

  NNLayer layer2;
  layer2.activation = ActivationFunctions::logarithmicAct;
  layer2.derivative = ActivationFunctions::logarithmicDrv;
  layer2.neuronCount = 10;


  std::vector<NNLayer> layers = {layer0, layer1, layer2};


  NeuralNetwork nn(layers);


  for(int sample = 0; sample < 1000; sample++){
    std::vector<Matrix>
            inputs,
            targets;

    inputs = targets = {};


    for(int r = 0; r < 10; r++){
      auto randomIndex = rand() % 50000;
      auto oldI = *helper.getInputAtIndex(randomIndex);
      Matrix in(PIXEL_COUNT, 1);
      unsigned int count = 0;
      for(unsigned int rows = 0; rows < oldI.rowCount(); rows++){
        for(unsigned int cols = 0; cols < oldI.colCount(); cols++){
          in.set(count, 0, oldI.get(rows, cols));
          count++;
        }
      }

      inputs.push_back(in);
      targets.push_back(*(helper.getTargetAtIndex(randomIndex)));

    }
    nn.train(inputs, targets);

  }







  return 0;
}




















void nnDemo(unsigned int epochs){
  MNIST helper;

  helper.ReadMNIST(POPULATION, "train-images-idx3-ubyte");
  helper.ReadMNISTLabels(POPULATION, "train-labels-idx1-ubyte");

  const unsigned int LAYER_COUNT = 3;

  unsigned int layers[LAYER_COUNT] = {PIXEL_COUNT,300, 10};

  SimpleNeuralNetwork imageProcessor(layers, 3, epochs);

  for(int trainPasses = 0; trainPasses < epochs; trainPasses++){
    std::cout << "On Training Pass " << trainPasses + 1 << std::endl;
    nnTrainer(imageProcessor, helper);
    std::cout << "Calculating Error Information" << std::endl;
    nnTester(imageProcessor, helper);
    std::cout << "\n\n\n";
    imageProcessor.addCurrentWeightsToFile();
  }
}







void nnTrainer(SimpleNeuralNetwork &nn, MNIST &helper){
  for(int sample = 0; sample < 1000; sample++){
    Matrix
            inputs[10],
            targets[10];


    for(int r = 0; r < 10; r++){
      auto randomIndex = rand() % 50000;
      auto oldI = *helper.getInputAtIndex(randomIndex);
      Matrix in(PIXEL_COUNT, 1);
      unsigned int count = 0;
      for(unsigned int rows = 0; rows < oldI.rowCount(); rows++){
        for(unsigned int cols = 0; cols < oldI.colCount(); cols++){
          in.set(count, 0, oldI.get(rows, cols));
          count++;
        }
      }

      inputs[r] = in;
      targets[r] = *(helper.getTargetAtIndex(randomIndex));

    }
    nn.trainWithSets(inputs, targets, 10);

  }
};




void nnTester(SimpleNeuralNetwork &digitProcessor, MNIST &helper){
  unsigned int guesses[10] = {0};
  unsigned int actuals[10] = {0};
  unsigned int correct[10] = {0};

  int errors = 0;
  for(unsigned int test = 50000; test < 60000; test++){
    auto firstData = helper.getInputAtIndex(test);
    Matrix in(firstData->rowCount() * firstData->colCount(), 1);
    unsigned int count = 0;
    for(unsigned int i = 0; i < firstData->rowCount(); i++){
      for(unsigned int j = 0; j < firstData->colCount(); j++){
        in.set(count, 0, firstData->get(i, j));
        count++;
      }
    }
    Matrix input = in;
    auto result = digitProcessor.processImage(input);

    guesses[result.getLargestComponentIndexInColumnVector()]++;
    actuals[helper.getTargetAtIndex(test)->getLargestComponentIndexInColumnVector()]++;

    if(result.getLargestComponentIndexInColumnVector() != helper.getTargetAtIndex(test)->getLargestComponentIndexInColumnVector()){
      errors++;
    } else {
      correct[result.getLargestComponentIndexInColumnVector()]++;
    }



  }
  std::cout << "\nTotal Errors: " << errors << " Out of 10000 or " << (1 - double(errors)/10000) * 100 << "% accuracy" << std::endl;

  int ratioSum = 0;
  for(int index = 0; index < 10; index++){
    int ratio;
    if(guesses[index] == 0){
      ratio = 0;
    } else {
      ratio = int(double(correct[index])/guesses[index] * 100);
    }
    std::cout << std::setw(10) << "Digit: " << std::setw(3) << index
              << std::setw(12) << "Expected: " << std::setw(10) << actuals[index]
              << std::setw(10) << "Found:" << std::setw(10) << guesses[index]
              << std::setw(10) << "Correct:" << std::setw(10) << correct[index]
              << std::setw(10) << "Ratio:" << std::setw(10) << ratio << "%"
              << std::endl;

    ratioSum += ratio;
  }

  std::cout << std::setw(10) << "\nAvg Ratio: " << std::setw(10) << ratioSum / 10 << "%";


}











void perceptronDemo(){
  float m, b;
  std::cout << "Please enter the slope of the line: ";
  std::cin >> m;
  std::cout << "Please enter the y-intercept of the line: ";
  std::cin >> b;


  Perceptron myFirstNeuron(2);
  Trainer myTrainer(myFirstNeuron, 2, 100000, m, b);

  float x,y;

  std::cout << "Please enter the x coordinate to test: ";
  std::cin >> x;
  std::cout << "Please enter the y coordinate to test: ";
  std::cin >> y;



  float testcase1[] = {x,y};

  auto result = myFirstNeuron.processInput(testcase1);
  if(result == 1){
    std::cout << "This is above the line" << std::endl;
  } else {
    std::cout << "This is below the line" << std::endl;
  }
}