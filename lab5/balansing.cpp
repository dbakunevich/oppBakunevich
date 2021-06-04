#include "balansing.h"
#include "mpi.h"
#include <cstdlib>

void *loadBalancing(void *args) {
    Args *arguments =  static_cast<Args *>(args);

    Task *list = arguments->list;
    MPI_Datatype MPI_ACK_Task_List = arguments->MPI_ACK_Task_List;
    pthread_mutex_t mutex = arguments->mutex;
    int size = arguments->size;
    int currentTask = arguments->currentTask;
    int listSize = arguments->listSize;

    int message = TURN_ON;
    while (message != TURN_OFF) {
        MPI_Status status;
        MPI_Recv(&message, 1, MPI_INT, MPI_ANY_SOURCE, ASK_TAG, MPI_COMM_WORLD, &status);

        if (message == ASK_FOR_TASK) {
            ACK_Task_List ackTaskList;
            pthread_mutex_lock(&mutex);
            if (currentTask >= listSize - 1) {
                pthread_mutex_unlock(&mutex);
                ackTaskList.ack.count = 0;
                MPI_Send(&ackTaskList, 1, MPI_ACK_Task_List, status.MPI_SOURCE, ACK_Task_List_TAG, MPI_COMM_WORLD);
            } else {
                double finishedFraction = currentTask / double(listSize);
                int taskCount = (listSize - currentTask) * finishedFraction / (size - 1) + 1;
                ackTaskList.ack.count = taskCount;

                Task *newList = new Task[taskCount];
                for (int i = 0; i < taskCount; ++i) {
                    newList[i].weight = list[listSize - taskCount + i].weight;
                }
                listSize -= taskCount;
                ackTaskList.list = newList;
                MPI_Send(&ackTaskList, 1, MPI_ACK_Task_List, status.MPI_SOURCE, ACK_Task_List_TAG, MPI_COMM_WORLD);
                pthread_mutex_unlock(&mutex);
            }
        }
    }
    return NULL;
}