#include <iostream>
#include <pthread.h>
#include <cstdlib>
#include <sys/types.h>
#include "mpi.h"
#include "execute.h"
#include "balansing.h"


void fillArgs(Args * args, Task *list, MPI_Datatype MPI_TASK,
              MPI_Datatype MPI_ACK, MPI_Datatype MPI_ACK_Task_List,
              pthread_mutex_t &mutex, int size, int rank, int startWeight,
              int startSize, int iterCount, int listSize) {

    args->list = list;
    args->MPI_TASK = MPI_TASK;
    args->MPI_ACK = MPI_ACK;
    args->MPI_ACK_Task_List = MPI_ACK_Task_List;
    args->mutex = mutex;
    args->size = size;
    args->rank = rank;
    args->startWeight = startWeight;
    args->startSize = startSize;
    args->iterCount = iterCount;
    args->currentTask = 0;
    args->listSize = listSize;
}

void createTypes(MPI_Datatype &MPI_TASK, MPI_Datatype &MPI_ACK,
                 MPI_Datatype &MPI_ACK_Task_List) {
    int blockLengths[1] = {1};
    MPI_Aint displacements[1];
    displacements[0] = 0;
    MPI_Datatype types[] = {MPI_INT};

    MPI_Type_create_struct(1, blockLengths, displacements, types, &MPI_TASK);
    MPI_Type_commit(&MPI_TASK);

    MPI_Type_create_struct(1, blockLengths, displacements, types, &MPI_ACK);
    MPI_Type_commit(&MPI_ACK);

    MPI_Type_create_struct(1, blockLengths, displacements, types, &MPI_ACK_Task_List);
    MPI_Type_commit(&MPI_ACK_Task_List);
}

int main(int argc, char *argv[]) {
    int provided;
    int error = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);


    Task *list = NULL;
    MPI_Datatype MPI_TASK, MPI_ACK, MPI_ACK_Task_List;
    pthread_mutex_t mutex;

    int size, rank;

    int startWeight;
    int startSize;
    int iterCount;

    int listSize = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided != MPI_THREAD_MULTIPLE) {
        if (!rank) {
            char errorString[MPI_MAX_ERROR_STRING];
            int len;
            MPI_Error_string(error, errorString, &len);
            std::cout << "ERROR: provided is not MPI_THREAD_MULTIPLE, provided: error code :"
                      << error << " - " << errorString << " provided: " << provided << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    if (argc < 4) {
        if (!rank) {
            std::cout << "Usage: " << argv[0] <<" iter_count list_size start_weight" << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    iterCount = atoi(argv[1]);
    startSize = atoi(argv[2]);
    startWeight = atoi(argv[3]);

    createTypes(MPI_TASK, MPI_ACK,
            MPI_ACK_Task_List);

    pthread_mutex_init(&mutex, nullptr);
    pthread_attr_t attributes;
    if (pthread_attr_init(&attributes) != 0) {
        std::cout << "ERROR: Cannot init attributes: " << errno << std::endl;
        MPI_Finalize();
        return 0;
    }
    pthread_t threads[2];
    Args* args = new Args;
    fillArgs(args, list, MPI_TASK, MPI_ACK, MPI_ACK_Task_List,
             mutex, size, rank, startWeight,
             startSize, iterCount, listSize);

    double start = MPI_Wtime();
    pthread_create(&threads[0], &attributes, loadBalancing, (void *) args);
    pthread_create(&threads[1], &attributes, processList, (void *) args);
    for (pthread_t thread : threads) {
        if (pthread_join(thread, nullptr) != 0) {
            std::cout << "ERROR: Cannot join a thread: " << errno << std::endl;
            MPI_Finalize();
            return 0;
        }
    }
    double end = MPI_Wtime();

    if (!rank) {
        std::cout << "Time: " << end - start << std::endl;
    }
    pthread_attr_destroy(&attributes);
    pthread_mutex_destroy(&mutex);
    free(list);
    delete args;

    MPI_Finalize();
    return 0;
}
