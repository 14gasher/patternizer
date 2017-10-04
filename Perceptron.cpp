//
// Created by Asher Gunsay on 9/18/17.
//

#include "Perceptron.hpp"

Perceptron::Perceptron(unsigned int inputCount)
{
  inputAndWeightCount = inputCount;
  weights = new float[inputCount];
  for (int i = 0; i < inputCount; i++)
  {
    weights[i] = -1 + static_cast<float>(rand()) / static_cast<float> (RAND_MAX / (2));
  }
}

Perceptron::~Perceptron()
{
  delete[] weights;
  weights = nullptr;
}

int Perceptron::processInput(float *inputs)
{
  float sum = 0;
  for (int i = 0; i < inputAndWeightCount; i++)
  {
    sum += inputs[i] * weights[i];
  }
  if (sum > 0)
  {
    return 1;
  } else
  {
    return -1;
  }
}

bool Perceptron::train(float *trainInputs, int desired)
{
  int guess = processInput(trainInputs);

  float error = desired - guess;
  for (int i = 0; i < inputAndWeightCount; i++)
  {
    weights[i] += c * error * trainInputs[i];
  }
  return !(error < 0.00001 && error > -0.00001);
}

Trainer::Trainer(Perceptron &p, unsigned int inputCount, unsigned int desiredTrainingPoints, float m, float b)
{
  slope = m;
  yIntercept = b;
  inputs = new float[inputCount];
  std::cout << "\n\nWe are training now!" << std::endl;
  if (inputCount != 2)
  {
    std::cout << "We have not been built to handle this much data yet! We currently can have 2..." << std::endl;
  } else
  {
    int count = 0;
    for (int i = 0; i < desiredTrainingPoints; i++)
    {
      if (i % 1000 == 0 && i != 0)
      {
        std::cout << count << " Errors from: " << i - 999 << " to " << i << std::endl;
        count = 0;
      }
      for (int j = 0; j < inputCount; j++)
      {
        inputs[j] = static_cast<float>((rand() % 2000) - 1000);
      }
      bool success = p.train(inputs, linearAnswer(inputs[0], inputs[1]));
      if (success)
      { count++; }


    }

    std::cout << "\nTraining Complete!\n\n";
  }

}

int Trainer::linearAnswer(float x, float z)
{
  float y = slope * x + yIntercept;
  if (z > y) {
    return 1;
  } else {
    return -1;
  }
}

Trainer::~Trainer(){
  delete[]inputs;
  inputs = nullptr;
}


