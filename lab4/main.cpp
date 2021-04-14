#include <iostream>
#include "mpi.h"

#define a 1e5
#define e 1e-8

#define DX 2
#define DY 2
#define DZ 2

#define NX 1000
#define NY 1000
#define NZ 1000


double f (double x, double y, double z) {
    return x * x + y * y + z * z;
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

    auto leftBorder = new double[Z * Y];
    auto rightBorder = new double[Z * Y];

    const double hx = 2.0 / (NX - 1);
    const double hy = 2.0 / (NY - 1);
    const double hz = 2.0 / (NZ - 1);

    const double hx2 = hx * hx;
    const double hy2 = hy * hy;
    const double hz2 = hz * hz;
    const double factor = 1 / (2 / hx2 + 2 / hy2 + 2 / hz2 + a);

    for (int x = 0, localX = shift[procRank]; x < X; ++x, ++localX) {
        for (int y = 0; y < Y; y++) {
            for (int z = 0; z < Z; z++) {
                if ((localX == 0) || (y == 0) || (z == 0) || (localX == NX - 1) || (y == NY - 1) || (z == NZ - 1)) {
                    functionIterations[0][x * Y * Z + y * Z + z] = f((localX * hx) - 1, (y * hy) - 1, (z * hz) - 1);
                    functionIterations[1][x * Y * Z + y * Z + z] = f((localX * hx) - 1, (y * hy) - 1, (z * hz) - 1);
                }
            }
        }
    }

}
