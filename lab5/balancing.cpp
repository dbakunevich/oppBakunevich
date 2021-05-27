#include "balancing.h"
#include "mpi.h"

void *loadBalancing(void *args) {
    BalansingArgs arg = *(BalansingArgs *) args;
    int message = TURN_ON;
    while (message != TURN_OFF) {
        MPI_Status status;
        MPI_Recv(&message, 1, MPI_INT, MPI_ANY_SOURCE, ASK_TAG, MPI_COMM_WORLD, &status);

        if (message == ASK_FOR_TASK) {
            ACK_Task_List ackTaskList;
            pthread_mutex_lock(arg.mutex);
            if (arg.currentTask >= arg.listSize - 1 || arg.gotTask) {
                pthread_mutex_unlock(arg.mutex);
                ackTaskList.ack.count = 0;
                MPI_Send(&ackTaskList, 1, *arg.MPI_ACK_Task_List, status.MPI_SOURCE, ACK_Task_List_TAG, MPI_COMM_WORLD);
            } else {
                double finishedFraction = *arg.currentTask / double(*arg.listSize);
                int taskCount = (arg.listSize - arg.currentTask) * finishedFraction / (*arg.size - 1) + 1;
                ackTaskList.ack.count = taskCount;

                auto *newList = new Task[taskCount];
                for (int i = 0; i < taskCount; ++i) {
                    newList[i].weight = arg.list[*arg.listSize - taskCount + i]->weight;
                }
                arg.listSize -= taskCount;
                ackTaskList.list = newList;
                MPI_Send(&ackTaskList, 1, *arg.MPI_ACK_Task_List, status.MPI_SOURCE, ACK_Task_List_TAG, MPI_COMM_WORLD);
                MPI_Send(ackTaskList.list, taskCount, *arg.MPI_TASK, status.MPI_SOURCE, TASK_TAG, MPI_COMM_WORLD);
                pthread_mutex_unlock(arg.mutex);
            }
        }
    }
    return nullptr;
}