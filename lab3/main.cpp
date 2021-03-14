#include <iostream>

#define N1 8
#define N2 8
#define N3 2

#define P1 8
#define P2 8

void fillMatrix(double * matrix, int firstBoard, int secondBoard){
    for (size_t i = 0; i < firstBoard; ++i) {
        for (size_t j = 0; j < secondBoard; ++j) {
            matrix[i * secondBoard + j] = 5.0f;
        }
    }
}

void matrixMul(const double * firstMatrix, const double * secondMatrix, double * resultMatrix){
    for (size_t i = 0; i < N1; ++i) {
        for (size_t j = 0; j < N3; ++j) {
            resultMatrix[i * N3+ j] = 0.0f;
            for (size_t k = 0; k < N2; ++k) {
                resultMatrix[i * N3 + j] += firstMatrix [i * N2 + k] * secondMatrix[k * N3 + j];
            }
        }
    }
}

void printMatrix(const double * matrix, int firstBoard, int secondBoard){
    for (size_t i = 0; i < firstBoard; ++i) {
        for (size_t j = 0; j < secondBoard; ++j) {
            printf("%f ", matrix[i * secondBoard + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    double *A = new double[N1 * N2];
    double *B = new double[N2 * N3];
    double *C = new double[N1 * N3];

    fillMatrix(A, N1, N2);
    fillMatrix(B, N2, N3);


    matrixMul(A, B, C);

    printMatrix(A, N1, N2);
    printMatrix(B, N2, N3);
    printMatrix(C, N1, N3);

    delete[] A;
    delete[] B;
    delete[] C;




}
