//
// Created by Asher Gunsay on 9/18/17.
//

#ifndef NEURALNETWORKS_MATRIX_HPP
#define NEURALNETWORKS_MATRIX_HPP


class Matrix
{
public:


  /**
   * The usual constructor
   *
   * @param rowCount
   * @param columnCount
   */
  Matrix(unsigned int rowCount, unsigned int columnCount);


  /**
   * Copy Constructor
   * @param oldMatrix
   */
  Matrix(const Matrix &oldMatrix);


  /**
   * Assignment operator
   * @param oldMatrix
   * @return
   */
  Matrix& operator=(const Matrix& oldMatrix);


  /**
   * Default constructor. Necessary for declaring arrays of matrices
   */
  Matrix(){};


  /**
   * Deconstructor. Cleans up the dynamic double array of matrix
   */
  ~Matrix();

  /**
   * Returns the transpose of a matrix
   *
   * for example, row matrix [ 1 & 1 ] becomes column matrix [ 1 \n 1 ]
   * @return Matrix transpose
   */
  Matrix transposition();


  /**
   * Adds matrices component wise.
   *
   * Matrices must be the same size.
   *
   * @param m
   * @return Sum of matrices
   */
  Matrix addition(Matrix& m);


  /**
   * Scalar multiplication of Matrices (scalar times each component)
   *
   * @param s scalar
   * @return Scaled matrix
   */
  Matrix scalar(double s);

  /**
   * Multiplies two matrices together.
   *
   * Matrix 1 must be m x n while Matrix 2 must be n x p and will return a Matrix of m x p
   * If this condition is not met, it will throw an error
   *
   * @param m
   * @return Matrix product
   */
  Matrix matrixMultiplation(Matrix& m);

  /**
   * Multiplies components of matrices together.
   * Matrices must be same size.
   * If not, error thrown.
   *
   * @param m
   * @return hadamard product
   */
  Matrix hadamardProduct(Matrix& m);

  /**
   * Only gets the kronecker product for vectors, but it works
   *
   * @param m
   * @return kronecker product
   */
  Matrix kroneckerVectorProduct(Matrix& m);

  /**
   * Appends one matrix to the end of another matrix.
   *
   * [1,1] augmented with [2,2] becomes [1,1,2,2]
   *
   * @param m
   * @return augmented matrix
   */
  Matrix matrixAugment(Matrix& m);

  /**
   * Return a matrix with all of the original matrice's components squared
   *
   * @return squared component matrix
   */
  Matrix squareComponents();

  /**
   * Add all of the componets and return the result
   * @return sum
   */
  double sumComponents();

  /**
   * Gets the largest index in a column vector
   *
   * @return index
   */
  int getLargestComponentIndexInColumnVector();

  /**
   * This prints out a representation of the matrix
   */
  void print();

  /**
   * @param i row index
   * @param j column index
   * @return value at given position
   */
  double get(unsigned int i, unsigned int j)const;

  /**
   * @param i row index
   * @param j row index
   * @param num value to set
   */
  void set(unsigned int i, unsigned int j, double num);

  unsigned int rowCount()const
  {return rows;};
  unsigned int colCount()const
  {return columns;};

  /**
   * Error classe
   */
  class outOfBounds{};

  /**
   * Error class
   */
  class sizesAreDifferent{};






private:
  unsigned int rows;
  unsigned int columns;
  double** matrix;

  /**
   * This allows us to know whether the matrix was allocated
   */
  bool allocated = false;

  /**
   * Cleans up memory
   */
  void pointerCleanUp();

  /**
   * Copies a matrix over properly
   *
   * @param oldMatrix
   * @return copy matrix
   */
  Matrix& copyMatrix(const Matrix& oldMatrix);

};


#endif //NEURALNETWORKS_MATRIX_HPP
