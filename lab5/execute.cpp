#include <iostream>
#include <cmath>
#include <cstdlib>
#include "execute.h"
#include "mpi.h"

Task *list;
MPI_Datatype MPI_ACK_Task_List;
pthread_mutex_t mutex;

int size, rank;

int currentTask;
int listSize;

bool gotTask;

int startWeight;
int startSize;
int iterCount;

int curIter = 0;
int tasksDone = 0;
long weightDone = 0;


void createList() {
    pthread_mutex_lock(&mutex);
    if (list != NULL) {
        delete (list);
    }
    list = new Task[startSize];
    listSize = startSize;

    currentTask = 0;

    for (int i = 0; i < startSize; ++i) {
        list[i].weight = startWeight + abs(50 - i % 100) * abs(rank - (curIter % size)) * startWeight;
    }
    pthread_mutex_unlock(&mutex);
}

int getTasks(int proc) {

    int message = ASK_FOR_TASK;
    MPI_Send(&message, 1, MPI_INT, proc, ASK_TAG, MPI_COMM_WORLD);

    ACK ack;
    Task * task;
    ACK_Task_List ackTaskList;
    MPI_Recv(&ackTaskList, 1, MPI_ACK_Task_List, proc, ACK_Task_List_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    ack = ackTaskList.ack;
    task = ackTaskList.list;

    int taskCount = ack.count;
    if (!taskCount) {
        return NO_TASK;
    } else {
        delete (list);
        list = new Task[taskCount];
        list = task;
        currentTask = 0;
        listSize = taskCount;
        gotTask = true;
        return TASK;
    }
}

long double completeTask(int weight) {
    long double globalRes = 0;
    for (int i = 0; i < weight; i++) {
        globalRes += sin(i);
    }
    return globalRes;
}

long double countListRes() {
    long double globalRes = 0;
    pthread_mutex_lock(&mutex);
    while (currentTask < listSize) {
        int weight = list[currentTask].weight;
        pthread_mutex_unlock(&mutex);

        weightDone += weight;
        globalRes = completeTask(weight);

        pthread_mutex_lock(&mutex);
        currentTask++;
        tasksDone++;
    }
    pthread_mutex_unlock(&mutex);
    return globalRes;
}

void *processList(void *args) {
    Args * arguments = static_cast<Args *> (args);

    list = arguments->list;
    MPI_ACK_Task_List = arguments->MPI_ACK_Task_List;
    mutex = arguments->mutex;
    size = arguments->size;
    rank = arguments->rank;
    currentTask = arguments->currentTask;
    listSize = arguments->listSize;
    startWeight = arguments->startWeight;
    startSize = arguments->startSize;
    iterCount = arguments->iterCount;

    gotTask = false;

    long double globalRes = 0;
    int lastReceivedTask = 0;
    while (curIter < iterCount) {
        if (!gotTask) {
            createList();
        }
        globalRes += countListRes();

        gotTask = false;
        for (int i = lastReceivedTask; i < size; i++) {
            if (i != rank) {
                if (getTasks(i) == TASK) {
                    lastReceivedTask = i;
                    break;
                }
            }
        }
        if (gotTask) {
            continue;
        }

        MPI_Barrier(MPI_COMM_WORLD);


        tasksDone = 0;
        weightDone = 0;
        globalRes = 0;
        lastReceivedTask = 0;
        ++curIter;
    }
    int message = TURN_OFF;
    MPI_Send(&message, 1, MPI_INT, rank, ASK_TAG, MPI_COMM_WORLD);
    return NULL;
}
