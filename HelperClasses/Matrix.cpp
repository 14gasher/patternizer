//
// Created by Asher Gunsay on 9/18/17.
//

#include "Matrix.hpp"
#include <string>
#include <iostream>
#include <iomanip>

Matrix::Matrix(unsigned int rowCount, unsigned int columnCount){
  rows = rowCount;
  columns = columnCount;
  allocated = true;
  matrix = new double*[rows];
  for(int i = 0; i < rows; i++){
    matrix[i] = new double[columns];
    for(int j = 0; j < columns; j++){
      matrix[i][j] = 0;
    }
  }
}
Matrix::Matrix(const Matrix &oldMatrix)
{
  copyMatrix(oldMatrix);
}

Matrix& Matrix::operator=(const Matrix &oldMatrix)
{
  // Clean up those pointers!
  pointerCleanUp();
  return copyMatrix(oldMatrix);
}

Matrix::~Matrix()
{
  pointerCleanUp();
}

Matrix& Matrix::copyMatrix(const Matrix &oldMatrix)
{
  rows = oldMatrix.rowCount();
  columns = oldMatrix.colCount();
  allocated = true;
  matrix = new double*[rows];
  for(unsigned int i = 0; i < rows; i++){
    matrix[i] = new double[columns];
    for(unsigned int j = 0; j < columns; j++){
      matrix[i][j] = oldMatrix.get(i,j);
    }
  }
  return *this;
}

void Matrix::pointerCleanUp()
{
  if(allocated){
    for(int i = 0; i < rows; i++){
      if(matrix[i] != NULL){
        delete[] matrix[i];
        matrix[i] = nullptr;
      }
    }
    delete[] matrix;
    matrix = nullptr;
  }
}



Matrix Matrix::transposition()
{
  Matrix transposed(columns, rows);
  for(unsigned int i = 0; i < rows; i++){
    for(unsigned int j = 0; j < columns; j++){
      transposed.set(j, i, matrix[i][j]);
    }
  }
  return transposed;
}

Matrix Matrix::addition(Matrix& m)
{
  if(rows != m.rowCount() || columns != m.colCount()){
    throw sizesAreDifferent();
  } else {
    Matrix added(rows, columns);
    for(unsigned int i = 0; i < rows; i++){
      for(unsigned int j = 0; j < columns; j++){
        added.set(i, j, matrix[i][j] + m.get(i,j));
      }
    }
    return added;
  }
}

Matrix Matrix::scalar(double s)
{
  Matrix scaled(rows, columns);
  for(unsigned int i = 0; i < rows; i++){
    for(unsigned int j = 0; j < columns; j++){
      scaled.set(i, j, s * matrix[i][j]);
    }
  }
  return scaled;
}

Matrix Matrix::matrixMultiplation(Matrix& m)
{
  if(columns != m.rowCount()){
    throw sizesAreDifferent();
  } else
  {
    Matrix product(rows, m.colCount());
    for (unsigned int i = 0; i < rows; i++)
    {
      for (unsigned int j = 0; j < m.colCount(); j++)
      {
        float res = 0.0;
        for (unsigned int k = 0; k < columns; k++)
        {
          res += matrix[i][k] * m.get(k, j);
        }
        product.set(i, j, res);
      }
    }
    return product;
  }

}

Matrix Matrix::hadamardProduct(Matrix& m)
{
  if(rows != m.rowCount() || columns != m.colCount()){
    throw sizesAreDifferent();
  } else {
    Matrix product(rows, columns);
    for(unsigned int i = 0; i < rows; i++){
      for(unsigned int j = 0; j < columns; j++){
        product.set(i, j, m.get(i,j) * matrix[i][j]);
      }
    }

    return product;
  }
}

Matrix Matrix::kroneckerVectorProduct(Matrix& m)
{
  if(columns != 1 || m.rowCount() != 1){
    throw sizesAreDifferent();
  }
  else{
    Matrix product(rows, m.colCount());
    for(unsigned int i = 0; i < rows; i++){
      for(unsigned int j = 0; j < m.colCount(); j++){
        product.set(i, j, matrix[i][0] * m.get(0, j));
      }
    }
    return product;
  }
}

Matrix Matrix::matrixAugment(Matrix& m)
{
  if(rows != m.rowCount()){
    throw sizesAreDifferent();
  } else {
    Matrix augmented(rows, columns + colCount());
    for(unsigned int i = 0; i < rows; i++){
      for(unsigned int j = 0; j < columns; j++){
        augmented.set(i, j, matrix[i][j]);
      }
    }
    for(unsigned int i = 0; i < rows; i++){
      for(unsigned int j = 0; j < m.colCount(); j++){
        augmented.set(i, j + columns,  m.get(i, j));
      }
    }
    return augmented;
  }


}

double Matrix::get (unsigned int i, unsigned int j)const
{
  if(i > rows || j > columns){
    throw outOfBounds();
  } else {
    return matrix[i][j];
  }

}
void Matrix::set(unsigned int i, unsigned int j, double num)
{
  if(i >= rows || j >= columns){
    this->print();
    throw outOfBounds();
  } else {
    matrix[i][j] = num;
  }
}

void Matrix::print(){
  std::cout<<std::endl;
  for(unsigned int i = 0; i < rows; i++){
    std::cout <<"| ";
    for(unsigned int j = 0; j < columns; j++){
       std::cout << std::setw(5) << matrix[i][j] << " ";
    }
    std::cout << "|" << std::endl;
  }
  std::cout<<std::endl;
}

double Matrix::sumComponents()
{
  double sum = 0.0;
  for(unsigned int i = 0; i < rows; i++)
  {
    for (unsigned int j = 0; j < columns; j++)
    {
      sum += matrix[i][j];
    }
  }
    return sum;
}

Matrix Matrix::squareComponents()
{
  Matrix square(rows, columns);
  for(unsigned int i = 0; i < rows; i++)
  {
    for (unsigned int j = 0; j < columns; j++)
    {
      square.set(i,j, matrix[i][j]*matrix[i][j]);
    }
  }
  return square;
}



int Matrix::getLargestComponentIndexInColumnVector()
{
  double max = matrix[0][0];
  int maxPos = 0;
  for(int i = 0; i < rows; i++){
    if(max < matrix[i][0]){
      max = matrix[i][0];
      maxPos = i;
    }

  }

  return maxPos;
}