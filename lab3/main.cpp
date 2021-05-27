#include <iostream>
#include <cstdlib>
#include <mpi.h>

/// A[N1 * N2], B[N2 * N3], C[N1 * N3]
#define N1 1024
#define N2 1024
#define N3 1024

/// Рандом для случайного заполнения матриц
/// А и В
#define MATRIX_MIN 0
#define MATRIX_MAX 1


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

void createMatrix(double ** A, double ** B, double ** C){
    *A = new double[N1 * N2];
    *B = new double[N2 * N3];
    *C = new double[N1 * N3];

    fillMatrix(*A, N1, N2);
    fillMatrix(*B, N2, N3);
}

void scatterA_B(double *A, double *segmentA, double *B, double *segmentB, const int *coords, int segmentRows, int segmentCols, MPI_Comm rowComm,
                MPI_Comm colComm){
    if (coords[0] == 0) {
        MPI_Scatter(A, segmentRows * N2, MPI_DOUBLE, segmentA, segmentRows * N2, MPI_DOUBLE, 0, colComm);
    }
    if (coords[1] == 0) {
        MPI_Datatype sendSegment;
        MPI_Datatype sendSegmentDouble;

        MPI_Type_vector(N2, segmentCols, N3, MPI_DOUBLE, &sendSegment);
        MPI_Type_commit(&sendSegment);

        MPI_Type_create_resized(sendSegment, 0, segmentCols * sizeof(double), &sendSegmentDouble);
        MPI_Type_commit(&sendSegmentDouble);

        MPI_Scatter(B, 1, sendSegmentDouble, segmentB, N2 * segmentCols, MPI_DOUBLE, 0, rowComm);

        MPI_Type_free(&sendSegment);
        MPI_Type_free(&sendSegmentDouble);
    }

    MPI_Bcast(segmentA, segmentRows * N2, MPI_DOUBLE, 0, rowComm);
    MPI_Bcast(segmentB, N2 * segmentCols, MPI_DOUBLE, 0, colComm);
}

void gatherC(double * C, double * segmentC, const int * dims, int * coords, int segmentRows, int segmentCols,
             int ProcNum, MPI_Comm gridComm, MPI_Comm rowComm, MPI_Comm colComm){
    MPI_Datatype recvSegment;
    MPI_Datatype recvSegmentDouble;

    MPI_Type_vector(segmentRows, segmentCols, N3, MPI_DOUBLE, &recvSegment);
    MPI_Type_commit(&recvSegment);

    MPI_Type_create_resized(recvSegment, 0, segmentCols * sizeof(double), &recvSegmentDouble);
    MPI_Type_commit(&recvSegmentDouble);

    int recvCounts[ProcNum];
    std::fill(recvCounts, recvCounts + ProcNum, 1);
    int displs[ProcNum];
    for (int procRank = 0; procRank < ProcNum; ++procRank) {
        MPI_Cart_coords(gridComm, procRank, 2, coords);
        displs[procRank] = dims[0] * segmentRows * coords[1] + coords[0];
    }

    MPI_Gatherv(segmentC, segmentRows * segmentCols, MPI_DOUBLE, C, recvCounts, displs, recvSegmentDouble,
                0, gridComm);

    MPI_Type_free(&recvSegment);
    MPI_Type_free(&recvSegmentDouble);

    MPI_Comm_free(&gridComm);
    MPI_Comm_free(&colComm);
    MPI_Comm_free(&rowComm);
}

void  mainWork(double * A, double  * B, double * C, int ProcNum, int ProcRank, const int * dims, int * coords, MPI_Comm gridComm, MPI_Comm rowComm, MPI_Comm colComm){
    int segmentRows = N1 / dims[1];
    int segmentCols = N3 / dims[0];
    double *segmentA = new double[segmentRows * N2];
    double *segmentB = new double[N2 * segmentCols];
    double *segmentC = new double[segmentRows * segmentCols];
    std::fill(segmentC, segmentC + segmentRows * segmentCols, 0);

    double matMulTime = -MPI_Wtime();

    scatterA_B(A, segmentA, B, segmentB, coords, segmentRows, segmentCols, rowComm, colComm);


    for (int i = 0; i < segmentRows; ++i) {
        for (int k = 0; k < N2; ++k) {
            for (int j = 0; j < segmentCols; ++j) {
                segmentC[i * segmentCols + j] += segmentA[i * N2 + k] * segmentB[k * segmentCols + j];
            }
        }
    }

    gatherC(C, segmentC, dims, coords, segmentRows, segmentCols,
            ProcNum, gridComm, rowComm, colComm);

    matMulTime += MPI_Wtime();

    if (ProcRank == 0) {
        std::cout << "matMulTime: " << matMulTime << "sec" << std::endl;
    }

    delete[] segmentA;
    delete[] segmentB;
    delete[] segmentC;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm gridComm;
    MPI_Comm rowComm;
    MPI_Comm colComm;

    int dims[2] = {4, 4};
    int periods[2] = {0, 0};
    int coords[2];
    int reorder = 0;
    int ProcNum, ProcRank;

    MPI_Comm_size (MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank (MPI_COMM_WORLD, &ProcRank);

    MPI_Dims_create(ProcNum, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &gridComm);
    MPI_Cart_coords(gridComm, ProcRank, 2, coords);
    MPI_Comm_split(gridComm, coords[1], coords[0], &rowComm);
    MPI_Comm_split(gridComm, coords[0], coords[1], &colComm);

    double *A;
    double *B;
    double *C;

    if (ProcRank == 0) {
        createMatrix(&A, &B, &C);
    }

    mainWork(A, B, C, ProcNum, ProcRank, dims, coords, gridComm, rowComm, colComm);

    if (ProcRank == 0) {
        delete[] A;
        delete[] B;
        delete[] C;
    }
    MPI_Finalize();
    return 0;
}
