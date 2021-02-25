#include <iostream>
#include <cmath>
#include <chrono>
#include <mpi.h>

#define epsilon 10e-5

using namespace std::chrono;

double scalarVectorsMultiplication(const double * vector1, const double * vector2, int N) {
    double resultScalar = 0;
    for (int i = 0; i < N; i++) {
        resultScalar += vector1[i] * vector2[i];
    }
    return resultScalar;
}

void vectorsSub(const double *vector1, const double *vector2, double *resultVector, int N) {
    for (int i = 0; i < N; i++) {
        resultVector[i] = vector1[i] - vector2[i];
    }
}

void matrixAndVectorMul(const double *matrix, const double *vector1, double *resultVector, int size, int *shift, const int* countRowsAtProc, int ProcRank) {

    auto *tmp = new double[size];
    for(int i = 0; i < countRowsAtProc[ProcRank]; i++){
        tmp[i] = 0.0f;
        for (int j = 0; j < size; j++) {
            tmp[i] += matrix[i * size + j] * vector1[j];
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allgatherv(tmp, countRowsAtProc[ProcRank], MPI_DOUBLE, resultVector, countRowsAtProc, shift, MPI_DOUBLE, MPI_COMM_WORLD);
    delete[] tmp;
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

    u = new double [size];
    for(int i = 0; i < size; i++) {
        u[i] = cos(2 * M_PI * i / size);
    }

    b = new double [size];
    matrixAndVectorMul(A, u, b, size, shift, countRowsAtProc, ProcRank);

    x = new double [size];
    for(int i = 0; i < size; i++) {
        x[i] = 0.0f;
    }

    r = new double [size];
    z = new double [size];
    Ax = new double [size];
    Az = new double [size];
}

double finishCount(double *rVector, double *bVector, int N) {
    double result;
    double lenOfVec_r = 0;
    double lenOfVec_b = 0;
    for (int i = 0; i < N; i++) {
        lenOfVec_r += pow(rVector[i], 2);
    }
    for (int i = 0; i < N; i++) {
        lenOfVec_b += pow(bVector[i], 2);
    }
    result = sqrt(lenOfVec_r) / sqrt(lenOfVec_b);
    return result;
}

int main(int argc, char *argv[])
{
    int N = 10000;
    if (argc > 1){
        N = atoi(argv[1]);
    }

    MPI_Init(&argc, &argv);
    int ProcNum, ProcRank;
    MPI_Comm_size ( MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank ( MPI_COMM_WORLD, &ProcRank);

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



    double      alpha,
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
    vectorsSub(b, Ax, r, N);

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
    while (epsilon <= finishCount(r, b, N)) {
        /**
         * Скалярное произведение(r_n, r_n)
        */
        firstScalar = scalarVectorsMultiplication(r, r, N);

        /**
        * Скалярное произведение(Az_n, z_n)
        */
        matrixAndVectorMul(A, z, Az, N, shift, countRowsAtProc, ProcRank);
        secondScalar = scalarVectorsMultiplication(Az, z, N);

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
        secondScalar = scalarVectorsMultiplication(r, r, N);

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