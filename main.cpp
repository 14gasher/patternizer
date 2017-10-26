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
const double E = 2.71828182845904523536;

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


void train(NeuralNetwork &nn, MNIST &helper, double learningRate){

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
    nn.train(inputs, targets, learningRate);

  }

}

double test(NeuralNetwork &nn, MNIST &helper, bool verbose){

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
    auto result = nn.processImage(input);

    guesses[result.getLargestComponentIndexInColumnVector()]++;
    actuals[helper.getTargetAtIndex(test)->getLargestComponentIndexInColumnVector()]++;

    if(result.getLargestComponentIndexInColumnVector() != helper.getTargetAtIndex(test)->getLargestComponentIndexInColumnVector()){
      errors++;
    } else {
      correct[result.getLargestComponentIndexInColumnVector()]++;
    }



  }
  if(verbose)
  {
    std::cout << "\nTotal Errors: " << errors << " Out of 10000 or " << (1 - double(errors) / 10000) * 100
              << "% accuracy" << std::endl;

    int ratioSum = 0;
    for (int index = 0; index < 10; index++)
    {
      int ratio;
      if (guesses[index] == 0)
      {
        ratio = 0;
      } else
      {
        ratio = int(double(correct[index]) / guesses[index] * 100);
      }
      std::cout << std::setw(10) << "Digit: " << std::setw(3) << index
                << std::setw(12) << "Expected: " << std::setw(10) << actuals[index]
                << std::setw(10) << "Found:" << std::setw(10) << guesses[index]
                << std::setw(10) << "Correct:" << std::setw(10) << correct[index]
                << std::setw(10) << "Ratio:" << std::setw(10) << ratio << "%"
                << std::endl;

      ratioSum += ratio;
    }

    std::cout << std::setw(10) << "\nAvg Ratio: " << std::setw(10) << ratioSum / 10 << "%" << std::endl;
  }

  return double(errors)/10000;

}


void modularNNDemo(MNIST &helper, NeuralNetwork &nn){

  double learningRate = 3;
  double smallestError = 1;
  unsigned int countDown = 0;
  unsigned int breakOutLimit = 50;
  unsigned int epoch = 0;

  while(countDown < breakOutLimit){
    std::cout << "On epoch " << epoch << " with learning rate " << learningRate << std::endl;
    train(nn, helper, learningRate);
    auto error = test(nn, helper, false);
    learningRate = learningRate - (1 - error) * learningRate / 10;
    if(learningRate < error){
      learningRate = error;
    }


    if(smallestError > error){
      smallestError = error;
      countDown = 0;

      std::cout << "    new best error: " << error << std::endl;
    } else {
      countDown++;
    }
    epoch++;

//  nn.printWeights();

  }

  std::cout << "\n\nBest error: " << smallestError << std::endl;

}










Matrix generateInputFromOutput(Matrix in1, Matrix in2){
  Matrix inLast(in1.rowCount() + in2.rowCount(),1);
  for(unsigned int rows = 0; rows < inLast.rowCount(); rows++){
    if(rows < in1.rowCount()){
      inLast.set(rows, 0, in1.get(rows,0));
    }else{
      inLast.set(rows, 0, in2.get(rows - in1.rowCount(), 0));
    }
  }
  return inLast;
}




void trainNetworkOfNetworks(MNIST &helper, double learningRate, NeuralNetwork &nn1, NeuralNetwork &nn2, NeuralNetwork &nn3){
  for(int sample = 0; sample < 1000; sample++){
    std::vector<Matrix>
            inputs,
            inputs2,
            targets;

    inputs = targets = inputs2 = {};


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
    nn1.train(inputs, targets, learningRate);
    nn2.train(inputs, targets, learningRate);


    for(int r = 0; r < 10; r++){
      auto in1 = nn1.processImage(inputs[0]);
      auto in2 = nn2.processImage(inputs[0]);

      inputs2.push_back(generateInputFromOutput(in1, in2));
    }
    nn3.train(inputs2, targets, learningRate);


  }



}

double testNetworkOfNetworks(MNIST &helper,  NeuralNetwork &nn1, NeuralNetwork &nn2, NeuralNetwork &nn3){


  int errors1 = 0;
  int errors2 = 0;
  int errors3 = 0;
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
    auto in1 = nn1.processImage(input);
    auto in2 = nn2.processImage(input);
    auto finalInput = generateInputFromOutput(in1,in2);

    auto result = nn3.processImage(finalInput);


    if(in1.getLargestComponentIndexInColumnVector() != helper.getTargetAtIndex(test)->getLargestComponentIndexInColumnVector()){
      errors1++;
    }
    if(in2.getLargestComponentIndexInColumnVector() != helper.getTargetAtIndex(test)->getLargestComponentIndexInColumnVector()){
      errors2++;
    }
    if(result.getLargestComponentIndexInColumnVector() != helper.getTargetAtIndex(test)->getLargestComponentIndexInColumnVector()){
      errors3++;
    }



  }

  std::cout << std::setw(20) << "        First errors: " << std::setw(7) << (errors1) << std::endl;
  std::cout << std::setw(20) << "        Second errors: " << std::setw(7) << (errors2) << std::endl;
  std::cout << std::setw(20) << "        Third errors: " << std::setw(7) << (errors3) << std::endl;


//  nn2.printWeights();

//  std::cout << "        Breakdown of Second Errors:" << std::endl;
//  test(nn2, helper, true);


  return double(errors3)/10000;


}






double newAct(double x){
  return log(1 + exp(E * (x - 1)));
}
double newDrv(double x){
  return E / (1 + exp(E * (x - 1)));
}


































int main()
{
  srand(time(NULL));
//
//  nnDemo(30);


  MNIST helper;

  helper.ReadMNIST(POPULATION, "train-images-idx3-ubyte");
  helper.ReadMNISTLabels(POPULATION, "train-labels-idx1-ubyte");


  NNLayer layerIn;
  layerIn.activation = ActivationFunctions::logarithmicAct;
  layerIn.derivative = ActivationFunctions::logarithmicDrv;
  layerIn.neuronCount = 28 * 28;

  NNLayer layerH1;
  layerH1.activation = ActivationFunctions::logarithmicAct;
  layerH1.derivative = ActivationFunctions::logarithmicDrv;
  layerH1.neuronCount = 20;


  NNLayer layerOut;
  layerOut.activation = ActivationFunctions::logarithmicAct;
  layerOut.derivative = ActivationFunctions::logarithmicDrv;
  layerOut.neuronCount = 10;


  std::vector<NNLayer> layers = {layerIn, layerH1, layerOut};




  NNLayer layerIn2;
  layerIn2.activation = ActivationFunctions::logarithmicAct;
  layerIn2.derivative = ActivationFunctions::logarithmicDrv;
  layerIn2.neuronCount = 28 * 28;

  NNLayer layerH12;
  layerH12.activation = ActivationFunctions::logarithmicAct;
  layerH12.derivative = ActivationFunctions::logarithmicDrv;
  layerH12.neuronCount = 20;


  NNLayer layerOut2;
  layerOut2.activation = ActivationFunctions::logarithmicAct;
  layerOut2.derivative = ActivationFunctions::logarithmicDrv;
  layerOut2.neuronCount = 10;


  std::vector<NNLayer> layers2 = {layerIn2, layerH12, layerOut2};




  NNLayer layerIn3;
  layerIn3.activation = ActivationFunctions::logarithmicAct;
  layerIn3.derivative = ActivationFunctions::logarithmicDrv;
  layerIn3.neuronCount = 20;

  NNLayer layerH13;
  layerH13.activation = ActivationFunctions::logarithmicAct;
  layerH13.derivative = ActivationFunctions::logarithmicDrv;
  layerH13.neuronCount = 20;


  NNLayer layerOut3;
  layerOut3.activation = ActivationFunctions::logarithmicAct;
  layerOut3.derivative = ActivationFunctions::logarithmicDrv;
  layerOut3.neuronCount = 10;


  std::vector<NNLayer> layers3 = {layerIn3, layerH13, layerOut3};







  NNLayer layerIn4;
  layerIn4.activation = ActivationFunctions::logarithmicAct;
  layerIn4.derivative = ActivationFunctions::logarithmicDrv;
  layerIn4.neuronCount = 28 * 28;

  NNLayer layerH14;
  layerH14.activation = ActivationFunctions::logarithmicAct;
  layerH14.derivative = ActivationFunctions::logarithmicDrv;
  layerH14.neuronCount = 20;

  NNLayer layerH24;
  layerH24.activation = ActivationFunctions::logarithmicAct;
  layerH24.derivative = ActivationFunctions::logarithmicDrv;
  layerH24.neuronCount = 20;

  NNLayer layerH34;
  layerH34.activation = ActivationFunctions::logarithmicAct;
  layerH34.derivative = ActivationFunctions::logarithmicDrv;
  layerH34.neuronCount = 20;


  NNLayer layerOut4;
  layerOut4.activation = ActivationFunctions::logarithmicAct;
  layerOut4.derivative = ActivationFunctions::logarithmicDrv;
  layerOut4.neuronCount = 10;


  std::vector<NNLayer> deepLayers = {layerIn4, layerH14, layerH24, layerH34, layerOut4};












  NeuralNetwork nn(layers);
  NeuralNetwork nn2(layers2);
  NeuralNetwork nn3(layers3);

  NeuralNetwork deepNetwork(deepLayers);

  double learningRate = 3;
  double smallestError = 1;
  unsigned int countDown = 0;
  unsigned int breakOutLimit = 50;



  double learningRateDeep = 3;
  double smallestErrorDeep = 1;
  unsigned int countDownDeep = 0;
  unsigned int breakOutLimitDeep = 50;


  unsigned int epoch = 0;


  while(countDown < breakOutLimit || countDownDeep < breakOutLimitDeep){
    std::cout << "On epoch " << epoch << " with learning rate " << learningRate << std::endl << std::endl;
    trainNetworkOfNetworks(helper, learningRate, nn, nn2, nn3);
    auto error = testNetworkOfNetworks(helper, nn, nn2, nn3);
    learningRate = learningRate - (1 - error) * learningRate / 10;
    if(learningRate < error){
      learningRate = error;
    }


    if(smallestError > error){
      smallestError = error;
      countDown = 0;

      std::cout << "    ------  new best Network of Networks' error: " << error << "  ------" <<std::endl << std::endl;
    } else {
      countDown++;
    }




    std::cout << "\n        Deep Learning Rate: " << learningRateDeep << std::endl;
    train(deepNetwork, helper, learningRateDeep);
    auto errorDeep = test(deepNetwork, helper, false);


    std::cout << "        Deep Errors: " << errorDeep << std::endl;
    if(learningRateDeep < errorDeep){
      learningRateDeep = errorDeep;
    }

    learningRateDeep = learningRateDeep - (1 - errorDeep) * learningRateDeep / 10;

    if(smallestErrorDeep > errorDeep){
      smallestErrorDeep = errorDeep;
      countDownDeep = 0;

      std::cout <<  "    >>>>>>  new best deep error: " << errorDeep << "  <<<<<<" <<std::endl << std::endl;
    } else {
      countDownDeep++;
    }



    epoch++;

//  nn.printWeights();

  }

  std::cout << "\n\nBest Network error: " << smallestError << std::endl;
  std::cout << "Best Deep error: " << smallestErrorDeep << std::endl;














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