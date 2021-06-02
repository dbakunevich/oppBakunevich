#include <iostream>
#include <cmath>
#include "execute.h"
#include "mpi.h"

void createList(pthread_mutex_t &mutex, Task * list, int startSize, int listSize,
                int &currentTask, int startWeight, int rank, int curIter, int size) {
    pthread_mutex_lock(&mutex);
    if (list != nullptr) {
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

int getTasks(int proc, Task * list, int rank, int &currentTask, bool &gotTask,
             MPI_Datatype MPI_ACK, MPI_Datatype MPI_ACK_Task_List, int &listSize) {

    int message = ASK_FOR_TASK;
    MPI_Send(&message, 1, MPI_INT, proc, ASK_TAG, MPI_COMM_WORLD);

    ACK ack;
    ACK_Task_List ackTaskList;
    MPI_Recv(&ackTaskList, 1, MPI_ACK_Task_List, proc, ACK_Task_List_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    ack = ackTaskList.ack;

    int taskCount = ack.count;
    if (!taskCount) {
        return NO_TASK;
    } else {
        delete (list);
        list = new Task[taskCount];
        std::cout << "Proc " << rank << " is getting " << taskCount << " tasks from " << proc << std::endl;
        MPI_Recv(list, taskCount, MPI_ACK, proc, TASK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        currentTask = 0;
        listSize = taskCount;
        gotTask = true;
        return TASK;
    }
}

double countListRes(pthread_mutex_t &mutex, Task *list, int listSize,
                    int &currentTask, long long &weightDone, int &tasksDone) {
    double globalRes = 0;
    pthread_mutex_lock(&mutex);
    while (currentTask < listSize) {
        int weight = list[currentTask].weight;
        pthread_mutex_unlock(&mutex);

        weightDone += weight;
        for (int i = 0; i < weight; i++) {
            globalRes += sin(i);
        }

        pthread_mutex_lock(&mutex);
        currentTask++;
        tasksDone++;
    }
    pthread_mutex_unlock(&mutex);
    return globalRes;
}

void *processList(void *args) {
    Args * arguments = static_cast<Args *> (args);

    Task *list = arguments->list;
    MPI_Datatype MPI_ACK = arguments->MPI_ACK;
    MPI_Datatype MPI_ACK_Task_List = arguments->MPI_ACK_Task_List;
    pthread_mutex_t mutex = arguments->mutex;
    int size = arguments->size;
    int rank = arguments->rank;
    int currentTask = arguments->currentTask;
    int listSize = arguments->listSize;
    bool gotTask = arguments->gotTask;


    int startWeight = arguments->startWeight;
    int startSize = arguments->startSize;
    int iterCount = arguments->iterCount;
    int curIter = arguments->curIter;
    int tasksDone = arguments->tasksDone;
    long long weightDone = arguments->weightDone;

    double globalRes = 0;
    double timeStart, timeEnd, duration;
    int lastReceivedTask = 0;
    while (curIter < iterCount) {
        if (!gotTask) {
            timeStart = MPI_Wtime();
            createList(mutex, list, startSize, listSize,
                    currentTask, startWeight, rank, curIter, size);
            printf("%d ls", listSize);

        }
        globalRes += countListRes(mutex, list, listSize,
                currentTask, weightDone, tasksDone);

        gotTask = false;
        for (int i = lastReceivedTask; i < size; i++) {
            if (i != rank) {
                if (getTasks(i, list, rank, currentTask, gotTask,
                        MPI_ACK, MPI_ACK_Task_List, listSize) == TASK) {
                    lastReceivedTask = i;
                    break;
                }
            }
        }
        if (gotTask) {
            continue;
        }

        timeEnd = MPI_Wtime();
        duration = timeEnd - timeStart;
        MPI_Barrier(MPI_COMM_WORLD);

        std::cout << "Proc " << rank << " finished " << curIter + 1
                  << " list_execute with " << tasksDone << " tasks and " << weightDone << " weight done" << std::endl;
        std::cout << "Proc " << rank << " global res is " << globalRes << " time spent on iteration "
                  << duration << std::endl;

        tasksDone = 0;
        weightDone = 0;
        globalRes = 0;
        lastReceivedTask = 0;
        ++curIter;
    }
    int message = TURN_OFF;
    MPI_Send(&message, 1, MPI_INT, rank, ASK_TAG, MPI_COMM_WORLD);
    return nullptr;
}
