//
// Created by Asher Gunsay on 9/18/17.
//

#ifndef NEURALNETWORKS_MATRIX_HPP
#define NEURALNETWORKS_MATRIX_HPP


class Matrix
{
public:
  // Usual constructor
  Matrix(unsigned int rowCount, unsigned int columnCount);
  // Copy constructor. Enables M1 = M2.
  Matrix(const Matrix &oldMatrix);
  Matrix& operator=(const Matrix& oldMatrix);
  // Default constructor. Necessary for declaring arrays of matrices
  Matrix(){};
  ~Matrix();
  Matrix transposition();
  Matrix addition(Matrix& m);
  Matrix scalar(double s);
  Matrix matrixMultiplation(Matrix& m);
  Matrix hadamardProduct(Matrix& m);

  Matrix kroneckerVectorProduct(Matrix& m);
  Matrix matrixAugment(Matrix& m);
  Matrix squareComponents();
  double sumComponents();
//  Matrix spliceCol(unsigned int col);
  int getLargestComponentIndexInColumnVector();

  Matrix returnSelf(){return *this;};

  void print();

  double get(unsigned int i, unsigned int j)const;
  void set(unsigned int i, unsigned int j, double num);
  unsigned int rowCount()const
  {return rows;};
  unsigned int colCount()const
  {return columns;};

  class outOfBounds{};
  class sizesAreDifferent{};






private:
  unsigned int rows;
  unsigned int columns;
  double** matrix;
  bool allocated = false;

  void pointerCleanUp();
  Matrix& copyMatrix(const Matrix& oldMatrix);

};


#endif //NEURALNETWORKS_MATRIX_HPP
