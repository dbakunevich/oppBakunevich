#ifndef EXECUTE_H
#define EXECUTE_H

#include <mpi.h>

#define ASK_TAG 1
#define TASK_TAG 3
#define ACK_Task_List_TAG 4
#define TASK 10
#define NO_TASK 11
#define ASK_FOR_TASK 12
#define TURN_OFF 13
#define TURN_ON 14

typedef struct Task {
    int weight;
} Task;

typedef struct ACK {
    int count;
} ACK;

typedef struct ACK_Task_List{
    ACK ack;
    Task *list;
}ACK_Task_List;

typedef struct Args{
    Task *list;
    MPI_Datatype MPI_TASK, MPI_ACK, MPI_ACK_Task_List;
    pthread_mutex_t mutex;

    int size, rank;

    int currentTask;
    int listSize;

    bool gotTask;

    int startWeight;
    int startSize;
    int iterCount;


} Args;

void *processList(void *args);

#endif