#include <iostream>
#include <cmath>
#include "mpi.h"

#define a 1e5
#define e 1e-8

#define DX 2.0f
#define DY 2.0f
#define DZ 2.0f

#define NX 40
#define NY 40
#define NZ 40


double f (double x, double y, double z) {
    return x * x + y * y + z * z;
}

double p(double x, double y, double z) {
    return 6 - a * f(x, y, z);
}

void fillFI(int X, int Y, int Z, double hx, double hy, double hz, int procRank,
            const int * shift, double *(functionIterations[])){
    for (int x = 0, localX = shift[procRank]; x < X; ++x, ++localX) {
        for (int y = 0; y < Y; y++) {
            for (int z = 0; z < Z; z++) {
                if ((localX == 0) || (y == 0) || (z == 0) || (localX == NX - 1) || (y == NY - 1) || (z == NZ - 1)) {
                    double phiValue = f((localX * hx) - 1, (y * hy) - 1, (z * hz) - 1);
                    functionIterations[0][x * Y * Z + y * Z + z] = phiValue;
                    functionIterations[1][x * Y * Z + y * Z + z] = phiValue;
                }
            }
        }
    }
}



double countElement(int x, int y, int z, int Y, int Z, double hx, double hy, double hz, double hx2, double hy2, double hz2,
             int procRank, int prevIter, double factor, const int *shift,
             const double *leftBorder, const double *rightBorder, int type, double *functionIterations[]) {
    double phix;
    if (type == 0) {
        phix = ((functionIterations[prevIter][(x - 1) * Y * Z + y * Z + z]) +
                       (functionIterations[prevIter][(x + 1) * Y * Z + y * Z + z])) / hx2;
    } else if (type == 1) {
        phix = (leftBorder[y * Z + z] +
                       (functionIterations[prevIter][(x + 1) * Y * Z + y * Z + z])) / hx2;
    } else{
        phix = ((functionIterations[prevIter][(x - 1) * Y * Z + y * Z + z]) +
                          rightBorder[y * Z + z]) / hx2;
    }

    double phiy = ((functionIterations[prevIter][x * Y * Z + (y - 1) * Z + z]) +
            (functionIterations[prevIter][x * Y * Z + (y + 1) * Z + z])) / hy2;
    double phiz = ((functionIterations[prevIter][x * Y * Z + y * Z + (z - 1)]) +
            (functionIterations[prevIter][x * Y * Z + y * Z + (z + 1)])) / hz2;
    return factor * (phix + phiy + phiz -
            p(((x + shift[procRank]) * hx) - 1, (y * hy) - 1, (z * hz) - 1));
}

double grade(int X, int Y, int Z, double hx, double hy, double hz, int procRank, int newIter,
             const int * shift, double *(functionIterations[])) {
    double max;
    double tmpMax = 0, abs;
    for (int x = 1; x < X - 1; ++x) {
        for (int y = 1; y < Y - 1; ++y) {
            for (int z = 1; z < Z - 1; ++z) {
                if ((abs = fabs(functionIterations[newIter][x * Y * Z + y * Z + z] -
                                f(((x + shift[procRank]) * hx) - 1, (y * hy) - 1, (z * hz) - 1))) > tmpMax) {
                    tmpMax = abs;
                }
            }
        }
    }
    MPI_Allreduce(&tmpMax, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    return max;

}

void waitBorders(int procRank, int procNum,
                 MPI_Request sendLeftBorder, MPI_Request sendRightBorder, MPI_Request recvLeftBorder, MPI_Request recvRightBorder){
    if (procRank != 0) {
        MPI_Wait(&sendLeftBorder, MPI_STATUS_IGNORE);
        MPI_Wait(&recvLeftBorder, MPI_STATUS_IGNORE);
    }
    if (procRank != procNum - 1) {
        MPI_Wait(&sendRightBorder, MPI_STATUS_IGNORE);
        MPI_Wait(&recvRightBorder, MPI_STATUS_IGNORE);
    }
}

void mainWork(int X, int Y, int Z, double hx, double hy, double hz, double factor, double hx2, double hy2, double hz2,
              int procRank, int procNum, const int * shift, const int * sizePerThreads,
              double * leftBorder, double * rightBorder, double *(functionIterations[])){

    fillFI(X, Y, Z, hx, hy, hz, procRank, shift, functionIterations);

    int newIter = 0, prevIter = 1;
    MPI_Request sendLeftBorder, sendRightBorder;
    MPI_Request recvLeftBorder, recvRightBorder;
    int criteria = 1;
    double time = -MPI_Wtime();


    while (criteria) {
        int tmpCriteria = 0;
        newIter = 1 - newIter;
        prevIter = 1 - prevIter;

        /// Send borders
        if (procRank != 0) {
            MPI_Isend(&(functionIterations[prevIter][0 + 0 + 0]), Y * Z, MPI_DOUBLE,
                      procRank - 1, 0, MPI_COMM_WORLD, &sendLeftBorder);
            MPI_Irecv(leftBorder, Y * Z, MPI_DOUBLE, procRank - 1, 1, MPI_COMM_WORLD, &recvLeftBorder);
        }
        if (procRank != procNum - 1) {
            MPI_Isend(&(functionIterations[prevIter][(sizePerThreads[procRank] - 1) * Y * Z + 0 + 0]), Y * Z, MPI_DOUBLE,
                      procRank + 1, 1, MPI_COMM_WORLD, &sendRightBorder);
            MPI_Irecv(rightBorder, Y * Z, MPI_DOUBLE, procRank + 1, 0, MPI_COMM_WORLD, &recvRightBorder);
        }

        /// Calculate subregions
        for (int x = 1; x < X - 1; ++x) {
            for (int y = 1; y < Y - 1; ++y) {
                for (int z = 1; z < Z - 1; ++z) {
                    double element = countElement(x, y, z, Y, Z, hx, hy, hz, hx2, hy2, hz2,
                                                  procRank, prevIter, factor, shift,
                                                  leftBorder, rightBorder, 0, functionIterations);
                    functionIterations[newIter][x * Y * Z + y * Z + z] = element;

                    tmpCriteria = fabs(element -
                                       f(((x + shift[procRank]) * hx) - 1, (y * hy) - 1, (z * hz) - 1)) > e ? 1 : 0;
                }
            }
        }

        /// Wait for borders
        waitBorders(procRank, procNum, sendLeftBorder, sendRightBorder, recvLeftBorder, recvRightBorder);

        /// Calculate borders
        for (int y = 1; y < Y - 1; ++y) {
            for (int z = 1; z < Z - 1; ++z) {
                /// Left border
                if (procRank != 0) {
                    int x = 0;
                    double element = countElement(x, y, z, Y, Z, hx, hy, hz, hx2, hy2, hz2,
                                                  procRank, prevIter, factor, shift,
                                                  leftBorder, rightBorder, 1, functionIterations);
                    functionIterations[newIter][x * Y * Z + y * Z + z] = element;

                    tmpCriteria = fabs(element -
                                       functionIterations[prevIter][x * Y * Z + y * Z + z]) > e ? 1 : 0;
                }

                /// Right border
                if (procRank != procNum - 1) {
                    int x = X - 1;

                    double element = countElement(x, y, z, Y, Z, hx, hy, hz, hx2, hy2, hz2,
                                                  procRank, prevIter, factor, shift,
                                                  leftBorder, rightBorder, 2, functionIterations);
                    functionIterations[newIter][x * Y * Z + y * Z + z] = element;

                    tmpCriteria = fabs(element -
                                       functionIterations[prevIter][x * Y * Z + y * Z + z]) > e ? 1 : 0;
                }
            }
        }
        MPI_Allreduce(&tmpCriteria, &criteria, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
    time += MPI_Wtime();
    double max = grade(X, Y, Z, hx, hy, hz, procRank, newIter, shift, functionIterations);

    if (procRank == 0) {
        std::cout << "Time: " << time << std::endl;
        std::cout << "Max difference: " << max << std::endl;
    }
}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int procNum, procRank;
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

    if (procRank == 0) {
        printf("ProcNum = %d\n Grid size: %d * %d * %d\n", procNum, NX, NY, NZ);
    }

    int sizePerThreads[procNum];
    int shift[procNum];

    std::fill(sizePerThreads, sizePerThreads + procNum, NX / procNum);

    for (int i = 0; i < NX % procNum; i++) {
        sizePerThreads[i]++;
    }

    shift[0] = 0;
    for (int i = 1; i < procNum; i++) {
        shift[i] = shift[i - 1] + sizePerThreads[i - 1];
    }

    int X = sizePerThreads[procRank];
    int Y = NY;
    int Z = NZ;

    double *(functionIterations[2]);
    functionIterations[0] = new double[X * Y * Z];
    functionIterations[1] = new double[X * Y * Z];
    std::fill(functionIterations[0], functionIterations[0] + X * Y * Z, 0);
    std::fill(functionIterations[1], functionIterations[1] + X * Y * Z, 0);

    double * leftBorder = new double[Z * Y];
    double * rightBorder = new double[Z * Y];

    const double hx = DX / (NX - 1);
    const double hy = DY / (NY - 1);
    const double hz = DZ / (NZ - 1);

    const double hx2 = hx * hx;
    const double hy2 = hy * hy;
    const double hz2 = hz * hz;
    const double factor = 1 / (2 / hx2 + 2 / hy2 + 2 / hz2 + a);

    mainWork(X, Y, Z, hx, hy, hz, factor, hx2, hy2, hz2, procRank, procNum, shift, sizePerThreads,
             leftBorder, rightBorder, functionIterations);


    delete[] functionIterations[0];
    delete[] functionIterations[1];
    delete[] leftBorder;
    delete[] rightBorder;

    MPI_Finalize();
    return 0;
}