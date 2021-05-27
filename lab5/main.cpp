#include <iostream>
#include <pthread.h>
#include "mpi.h"
#include "execute.h"
#include "balancing.h"

Task *list = nullptr;
MPI_Datatype MPI_TASK, MPI_ACK, MPI_ACK_Task_List;
pthread_mutex_t mutex;

int size, rank;

int startWeight;
int startSize;
int iterCount;
int curIter = 0;

int currentTask = 0;
int listSize;

int tasksDone = 0;
long long weightDone = 0;
bool gotTask = false;

typedef struct ExecuteArgs{
    Task *list;
    MPI_Datatype MPI_TASK, MPI_ACK, MPI_ACK_Task_List;
    pthread_mutex_t mutex;

    int size, rank;

    int startWeight;
    int startSize;
    int iterCount;
    int curIter;

    int currentTask;
    int listSize;

    int tasksDone;
    long long weightDone;
    bool gotTask;
}ExecuteArgs;

void fillArgs(BalansingArgs balansingArgs, ExecuteArgs executeArgs) {
    balansingArgs.list = &list;
    balansingArgs.MPI_TASK = &MPI_TASK;
    balansingArgs.MPI_ACK = &MPI_ACK;
    balansingArgs.MPI_ACK_Task_List = &MPI_ACK_Task_List;

    balansingArgs.size = &size;
    balansingArgs.rank = &rank;
    balansingArgs.currentTask = &currentTask;
    balansingArgs.listSize = &listSize;
    balansingArgs.gotTask = &gotTask;
}

void createTypes() {
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
            std::cout << "Usage: loadBalancing.exe iter_count list_size start_weight" << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    iterCount = atoi(argv[1]);
    startSize = atoi(argv[2]);
    startWeight = atoi(argv[3]);

    createTypes();

    pthread_mutex_init(&mutex, nullptr);
    pthread_attr_t attributes;
    if (pthread_attr_init(&attributes) != 0) {
        std::cout << "ERROR: Cannot init attributes: " << errno << std::endl;
        MPI_Finalize();
        return 0;
    }
    pthread_t threads[2];
    double start = MPI_Wtime();
    BalansingArgs balansingArgs;
    ExecuteArgs executeArgs;

    fillArgs(balansingArgs, executeArgs);


    pthread_create(&threads[0], &attributes, loadBalancing, (void*) &balansingArgs);
    pthread_create(&threads[1], &attributes, processList, nullptr);
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

    pthread_mutex_destroy(&mutex);
    MPI_Finalize();
    return 0;
}
