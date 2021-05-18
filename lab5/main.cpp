#include <iostream>
#include <pthread.h>
#include <cmath>
#include "mpi.h"

int size, rank;

int startWeight;
int startSize;
int iterCount;

int main(int argc, char *argv[]) {
    int provided;
    int error = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (!rank) {
            std::cout << "Usage: loadBalancing.exe iter_count list_size start_weight" << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    iterCount = atoi(argv[1]);
    startSize = atoi(argv[2]);
    startWeight = atoi(argv[3]);

    MPI_Finalize();
    return 0;
}
