#include <iostream>
#include <mpi.h>

/// A[N1 * N2], B[N2 * N3], C[N1 * N3]
#define N1 8
#define N2 4
#define N3 4

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

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm gridComm;
    MPI_Comm rowComm;
    MPI_Comm colComm;

    int dims[2] = {0, 0};
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


    double *A = new double[N1 * N2];
    double *B = new double[N2 * N3];
    double *C = new double[N1 * N3];

    if(ProcRank == 0) {
        fillMatrix(A, N1, N2);
        fillMatrix(B, N2, N3);
        //std::fill(C, C + N1 * N3, 0);
    }


    int segmentRows = N1 / dims[1];
    int segmentCols = N3 / dims[0];
    double *segmentA = new double[segmentRows * N2];
    double *segmentB = new double[N2 * segmentCols];
    double *segmentC = new double[segmentRows * segmentCols];
    std::fill(segmentC, segmentC + segmentRows * segmentCols, 0);

    double matMulTime = -MPI_Wtime();

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

    double segmentMulTime = -MPI_Wtime();
    for (int i = 0; i < segmentRows; ++i) {
        for (int k = 0; k < N2; ++k) {
            for (int j = 0; j < segmentCols; ++j) {
                segmentC[i * segmentCols + j] += segmentA[i * N2 + k] * segmentB[k * segmentCols + j];
            }
        }
    }
    segmentMulTime += MPI_Wtime();

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

    matMulTime += MPI_Wtime();

    if (ProcRank == 0) {
        std::cout << "segmentMulTime(0): " << segmentMulTime << "sec" << std::endl;
        std::cout << "matMulTime: " << matMulTime << "sec" << std::endl;

        printMatrix(A, N1, N2);
        printMatrix(B, N2, N3);
        printMatrix(C, N1, N3);
    }



    delete[] A;
    delete[] B;
    delete[] C;

    delete[] segmentA;
    delete[] segmentB;
    delete[] segmentC;

    MPI_Finalize();
    return 0;
}
