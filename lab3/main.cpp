#include <iostream>
#include <mpi.h>

/// A[N1 * N2], B[N2 * N3], C[N1 * N3]
#define N1 8
#define N2 8
#define N3 2

/// Решетка
#define P1 2
#define P2 2

/// Рандом для случайного заполнения матриц
/// А и В
#define MATRIX_MIN -50
#define MATRIX_MAX 50

double getRandomDouble(double min, double max){
    return (max - min) * ((double) rand() / (double)RAND_MAX) + min;
}

void fillMatrix(double * matrix, int firstBoard, int secondBoard){
    for (size_t i = 0; i < firstBoard; ++i) {
        for (size_t j = 0; j < secondBoard; ++j) {
            matrix[i * secondBoard + j] = getRandomDouble(MATRIX_MIN, MATRIX_MAX);
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

void matrixMul(double *firstMatrix, double *secondMatrix, double *resultMatrix, MPI_Comm GridComm) {

}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int ProcNum, ProcRank;
    MPI_Comm_size (MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank (MPI_COMM_WORLD, &ProcRank);


    MPI_Comm gridComm;
    MPI_Comm rowComm;
    MPI_Comm colComm;

    int dims[2] = {0, 0};
    int periods[2] = {0, 0};
    int coords[2] = {0, 1};
    int reorder = 0;
    MPI_Dims_create(ProcNum, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder,
                    &gridComm);
    MPI_Comm_rank (gridComm, &ProcRank);


    MPI_Cart_coords(gridComm, ProcRank, 2, coords);

    MPI_Comm_split(gridComm, coords[1], coords[0], &rowComm);
    MPI_Comm_split(gridComm, coords[0], coords[1], &colComm);



    double *A = new double[N1 * N2];
    double *B = new double[N2 * N3];
    double *C = new double[N1 * N3];

    if(ProcRank == 0) {
        fillMatrix(A, N1, N2);
        fillMatrix(B, N2, N3);
    }


    matrixMul(A, B, C, gridComm);

    printMatrix(A, N1, N2);
    printMatrix(B, N2, N3);
    printMatrix(C, N1, N3);

    delete[] A;
    delete[] B;
    delete[] C;
}
