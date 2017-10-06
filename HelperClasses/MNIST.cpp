//
// Created by Asher Gunsay on 9/23/17.
//

#include "MNIST.hpp"


unsigned int MNIST::ReverseInt (unsigned int i)
{
  unsigned char ch1, ch2, ch3, ch4;
  ch1=i&255;
  ch2=(i>>8)&255;
  ch3=(i>>16)&255;
  ch4=(i>>24)&255;
  return((unsigned int)ch1<<24)+((unsigned int)ch2<<16)+(( unsigned int)ch3<<8)+ch4;
}
void MNIST::ReadMNIST(unsigned int NumberOfImages, std::string filepath)
{
  numberOfImages = NumberOfImages;
  inputs = new Matrix*[NumberOfImages];
  std::ifstream file (filepath,std::ios::binary);
  if (file.is_open())
  {
    std::cout << "Processing input images" << std::endl;
    unsigned int magic_number=0;
    unsigned int number_of_images=0;
    unsigned int n_rows=0;
    unsigned int n_cols=0;
    file.read((char*)&magic_number,sizeof(magic_number));
    magic_number= ReverseInt(magic_number);
    file.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images= ReverseInt(number_of_images);
    file.read((char*)&n_rows,sizeof(n_rows));
    n_rows= ReverseInt(n_rows);
    file.read((char*)&n_cols,sizeof(n_cols));
    n_cols= ReverseInt(n_cols);
    for(int i=0;i<number_of_images;++i)
    {
      inputs[i] = new Matrix(n_rows, n_cols);
      for(unsigned int r=0;r<n_rows;++r)
      {
        for(unsigned int c=0;c<n_cols;++c)
        {
          unsigned char temp=0;
          file.read((char*)&temp,sizeof(temp));
          inputs[i]->set(r,c,((double)temp + 1 )/ 256.0);
        }
      }


    }
  } else {
    std::cout << "file not found" << std::endl;
  }
}

void MNIST::ReadMNISTLabels(unsigned int NumberOfImages, std::string filepath){
  numberOfImages = NumberOfImages;
  targets = new Matrix*[NumberOfImages];
  std::ifstream file (filepath,std::ios::binary);
  if (file.is_open())
  {
    std::cout << "Processing input labels" << std::endl;
    unsigned int magic_number=0;
    unsigned int number_of_images=0;

    file.read((char*)&magic_number,sizeof(magic_number));
    magic_number= ReverseInt(magic_number);
    file.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images= ReverseInt(number_of_images);

    for(unsigned int i=0;i<number_of_images;++i)
    {
      targets[i] = new Matrix(10, 1);
      unsigned char temp=0;
      file.read((char*)&temp,sizeof(temp));
      unsigned int answer = (unsigned int)temp;
      for(unsigned int j = 0; j<10; j++){
        targets[i]->set(j,0,0.0);
      }


      targets[i]->set(answer,0, 1);

    }
  } else {
    std::cout << "file not found" << std::endl;
  }
};

