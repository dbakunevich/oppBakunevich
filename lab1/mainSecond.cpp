#include <iostream>
#include <cmath>
#include <chrono>
#include <mpi.h>

#define epsilon 10e-5

using namespace std::chrono;

double finishCount(double *rVector, double *bVector, const int *countRowsAtProc, int ProcRank) {
    double result;
    double lenOfVec_r = 0.0f;
    double lenOfVec_b = 0.0f;
    for (int i = 0; i < countRowsAtProc[ProcRank]; i++) {
        lenOfVec_r += pow(rVector[i], 2);
    }
    for (int i = 0; i < countRowsAtProc[ProcRank]; i++) {
        lenOfVec_b += pow(bVector[i], 2);
    }

    double Vec_r = 0.0f;
    double Vec_b = 0.0f;

    MPI_Allreduce(&lenOfVec_r, &Vec_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&lenOfVec_b, &Vec_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    result = sqrt(Vec_r) / sqrt(Vec_b);
    return result;
}

double scalarVectorsMultiplication(const double *vector1, const double *vector2, const int *countRowsAtProc, int ProcRank) {
    double resultScalar = 0.0f;
    for (int i = 0; i < countRowsAtProc[ProcRank]; i++) {
        resultScalar += vector1[i] * vector2[i];
    }
    double scalar = 0.0f;
    MPI_Allreduce(&resultScalar, &scalar, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return resultScalar;
}

void vectorsSub(const double *vector1, const double *vector2, double *resultVector, const int* countRowsAtProc, int ProcRank) {
    for (int i = 0; i < countRowsAtProc[ProcRank]; i++) {
        resultVector[i] = vector1[i] - vector2[i];
    }
}

void matrixAndVectorMul(const double *matrix, const double *vector1, double *resultVector, int size, const int *shift, const int* countRowsAtProc, int ProcRank) {
    //считаем partA * partX - записываем в буфер
    auto * mulBuf = new double [countRowsAtProc[ProcRank] * size]; //буффер для умножения
    for (int k = 0; k < countRowsAtProc[ProcRank]*size; ++k)
        mulBuf[k] = 0.0f;

    for (int i = 0; i < countRowsAtProc[ProcRank]; ++i) {
        for (int j = 0; j < size; ++j) {
            mulBuf[i * size + j] += matrix[i * size + j] * vector1[i];
        }
    }
    //надо дебажить тут что-то не так с индексами массивов и умножением или суммой
    //построчно суммируем элементы буфера
    //так как при инициализации матрицу транспонировал, чтобы вычилсения были быстрее
    auto * buffer = new double [size];
    for (int k = 0; k < size; ++k)
        buffer[k] = 0.0f;

    for (int i = 0; i < countRowsAtProc[ProcRank]; ++i) {
        for (int j = 0; j < size; ++j) {
            buffer[j] += mulBuf[i * size + j];
        }
    }
    //теперь нужно проссумировать соответсвующии блоки в каждом процессе
    //каждый процесс отправляет нужную часть своего просуммированого столбца
    // ( с shift[i] по countRowsAtProc[i] процессу i)
    // процесс с номером i принимает эти данные и суммирует в соотвестующих яейках pratAx
    MPI_Allreduce( MPI_IN_PLACE, buffer, size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for(int i = 0; i < countRowsAtProc[ProcRank]; i++){
        resultVector[i] = buffer[i + shift[ProcRank]];
    }

    delete [] buffer;
    delete [] mulBuf;


    /*auto *allVector1 = new double [size];
    MPI_Allgather(vector1, countRowsAtProc[ProcRank], MPI_DOUBLE, allVector1, size, MPI_DOUBLE, MPI_COMM_WORLD);

    //auto *tmp = new double[size];
    for(int i = 0; i < countRowsAtProc[ProcRank]; i++) {
        resultVector[i] = 0.0f;
        for (int j = 0; j < size; j++) {
            resultVector[i] += matrix[i * size + j] * vector1[j];
        }
    }
    //MPI_Barrier(MPI_COMM_WORLD);
    //MPI_Allgatherv(tmp, countRowsAtProc[ProcRank], MPI_DOUBLE, resultVector, countRowsAtProc, shift, MPI_DOUBLE, MPI_COMM_WORLD);
    //delete[] tmp;*/
}

void init(double *&x, double *&A, double *&b, double *&u, double *&r, double *&z, double *&Ax, double *&Az, int size, int ProcRank, int ProcNum, int* &shift, int* &countRowsAtProc){
    shift = new int [ProcNum]; //смещение по элементам от начала для каждого процесса
    countRowsAtProc = new int [ProcNum]; // колво элементов которое процесс считает

    int remainder = size % ProcNum; //остаток
    int quotient = size / ProcNum; //частное

    int startLine;//номер строки с которой процесс считает матрицу A
    int NumberOfLines;//количествно строк которые обрабатывает каждый процесс
    //раскидываем по доп строке на процессы с рангом меньшим чем остаток(нумерация с 0)
    if(ProcRank < remainder) {
        NumberOfLines = quotient + 1;
        startLine = (ProcRank) * (NumberOfLines);

        A = new double [NumberOfLines * size];

        for(int i = 0; i < NumberOfLines; i++) {
            for(int j = 0; j < size; j++) {
                A[i * size + j]= 1.0f;
            }
            A[i * size + startLine + i] = 2.0f;
        }
    } else {
        NumberOfLines = quotient;
        startLine = (remainder * (NumberOfLines + 1)) + ((ProcRank - remainder) * NumberOfLines);

        A = new double [NumberOfLines * size];

        for (int i = 0; i < NumberOfLines; i++) {
            for(int j = 0; j < size; j++) {
                A[i * size + j] = 1.0f;
            }
            A[i * size + startLine + i] = 2.0f;
        }
    }

    /**
     * "синхронизируем данные" -> теперь каждый процес будет знать
     * количесвто строк которое исполняет другой процесс,
     * а так же место в матрице с которого он начинает исполнение
     * в ходе цикла каждый прцоесс отправит каждому своим данные
     */
    MPI_Status s;
    for (int i = 0; i < ProcNum; i++) {
        MPI_Sendrecv(&NumberOfLines, 1, MPI_INTEGER, i, 0, (countRowsAtProc+i), 1, MPI_INTEGER, i, 0, MPI_COMM_WORLD, &s);
        MPI_Sendrecv(&startLine, 1, MPI_INTEGER, i, 0, (shift+i), 1, MPI_INTEGER, i, 0, MPI_COMM_WORLD, &s);
    }

    u = new double [NumberOfLines];
    for(int i = 0; i < NumberOfLines; i++) {
        u[i] = cos(2 * M_PI * i / NumberOfLines);
    }

    b = new double [NumberOfLines];
    matrixAndVectorMul(A, u, b, NumberOfLines, shift, countRowsAtProc, ProcRank);

    x = new double [NumberOfLines];
    for(int i = 0; i < NumberOfLines; i++) {
        x[i] = 0.0f;
    }

    r = new double [NumberOfLines];
    z = new double [NumberOfLines];
    Ax = new double [NumberOfLines];
    Az = new double [NumberOfLines];
}

int main(int argc, char *argv[])
{
    int N = 10000;
    if (argc > 1){
        N = atoi(argv[1]);
    }

    MPI_Init(&argc, &argv);
    int ProcNum, ProcRank;
    MPI_Comm_size (MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank (MPI_COMM_WORLD, &ProcRank);

    double *A = nullptr;
    double *b = nullptr;
    double *x = nullptr;
    double *u = nullptr;
    double *r = nullptr;
    double *z = nullptr;
    double *Ax = nullptr;
    double *Az = nullptr;

    int *countRowsAtProc;
    int *shift;

    init(x, A, b, u, r, z, Ax, Az, N, ProcRank, ProcNum, shift, countRowsAtProc);



    double  alpha,
            beta,
            firstScalar,
            secondScalar;


    auto startTime = system_clock::now();

    /**
     * Далее у нас идёт чисто метод сопряжённых градиентов
     * r_0 = b - Ax_0
    */
    matrixAndVectorMul(A, x, Ax, N, shift, countRowsAtProc, ProcRank);
    for (int i = 0; i < N; i++) {
        r[i] = b[i];
    }
    vectorsSub(b, Ax, r, countRowsAtProc, ProcRank);

    /**
     * z_0 = r_0
    */
    for (int i = 0; i < N; i++) {
        z[i] = r[i];
    }


    /**
     * Выполняем итерации в цикле до тех пор, пока критерий
     * завершения счёта ||r_n|| / ||b|| не будет меньше заданной точности, именно
     * того эпсилон, который мы задаём сами, чем меньше - тем лучше
    */
    while (epsilon <= finishCount(r, b, countRowsAtProc, ProcRank)) {
        MPI_Barrier(MPI_COMM_WORLD);
        /**
         * Скалярное произведение(r_n, r_n)
        */
        firstScalar = scalarVectorsMultiplication(r, r, countRowsAtProc, ProcRank);

        /**
        * Скалярное произведение(Az_n, z_n)
        */
        matrixAndVectorMul(A, z, Az, N, shift, countRowsAtProc, ProcRank);
        secondScalar = scalarVectorsMultiplication(Az, z, countRowsAtProc, ProcRank);

        /**
         * alpha =(r_n, r_n)/(Az_n, z_n)
        */
        alpha = firstScalar / secondScalar;

        /**
         * x_n+1 = x_n + alpha_n+1*z_n
         * r_n+1 = r_n - alpha_n+1*A*z_n
        */
        for (int i = 0; i < N; i++) {
            x[i] += alpha * z[i];
            r[i] -= alpha * Az[i];
        }
        secondScalar = scalarVectorsMultiplication(r, r, countRowsAtProc, ProcRank);

        /**
         * beta =(r_n+1, r_n+1)/(r_n, r_n)
        */
        beta = secondScalar / firstScalar;

        /**
         * z_n+1 = r_n+1 + beta_n+1*z_n
        */
        for (int i = 0; i < N; i++) {
            z[i] = r[i] + (beta * z[i]);
        }
    }
    auto endTime = system_clock::now();
    auto duration = duration_cast<nanoseconds>(endTime - startTime);

    /**
     * Сравниваем наш полученный вектор x и наш эталонный вектор u
     * Ура, они одинаковые
     */
    std::cout << "Compare u[] and x[] is equals" << std::endl;
    std::cout << "Time: " << duration.count() / double(10e8) << " sec" << std::endl;

    delete[] A;
    delete[] b;
    delete[] x;
    delete[] u;
    delete[] r;
    delete[] z;
    delete[] Ax;
    delete[] Az;

    MPI_Finalize();
    return 0;
}