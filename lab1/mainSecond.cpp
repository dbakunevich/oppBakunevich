#include <iostream>
#include <cmath>
#include <mpi.h>

#define epsilon 10e-5


double dotProduct(const double *vector1, const double *vector2, int count) {
    double sum = 0;
    for (int i = 0; i < count; ++i) {
        sum += vector1[i] * vector2[i];
    }
    double fullSum;
    MPI_Allreduce(&sum, &fullSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return fullSum;
}

double finishCount(double *rVector, double *bVector, int countRowsAtProc) {
    double result;
    double lenOfVec_r = dotProduct(rVector, rVector, countRowsAtProc);
    double lenOfVec_b = dotProduct(bVector, bVector, countRowsAtProc);

    result = sqrt(lenOfVec_r) / sqrt(lenOfVec_b);
    return result;
}

double scalarVectorsMultiplication(const double *vector1, const double *vector2, int countRowsAtProc) {
    double resultScalar = 0.0f;
    for (int i = 0; i < countRowsAtProc; i++) {
        resultScalar += vector1[i] * vector2[i];
    }
    double scalar = 0.0f;
    MPI_Allreduce(&resultScalar, &scalar, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return scalar;
}

void vectorsSub(const double *vector1, const double *vector2, double *resultVector, int countRowsAtProc) {
    for (int i = 0; i < countRowsAtProc; i++) {
        resultVector[i] = vector1[i] - vector2[i];
    }
}

void matrixAndVectorMul(const double *matrix, const double *vector1, double *resultVector, int size, int *shift, const int* countRowsAtProc, int ProcNum, int ProcRank) {
    for(int i = 0; i < countRowsAtProc[ProcRank]; ++i){
        resultVector[i] = 0.0f;
        for (int j = 0; j < size; ++j) {
            resultVector[i] = matrix[i * size + j] * vector1[i];
        }
    }
}

void init(double *&x, double *&A, double *&b, double *&u, double *&r, double *&z, double *&Ax, double *&Az, int size, int ProcRank, int ProcNum, int *&shift, int *&countRowsAtProc){
    ///смещение по элементам от начала для каждого процесса
    shift = new int [ProcNum];
    /// колво элементов которое процесс считает
    countRowsAtProc = new int [ProcNum];

    ///остаток
    int remainder = size % ProcNum;
    ///частное
    int quotient = size / ProcNum;

    ///номер строки с которой процесс считает матрицу A
    int startLine;
    ///количествно строк которые обрабатывает каждый процесс
    int NumberOfLines;
    ///раскидываем по доп строке на процессы с рангом меньшим чем остаток(нумерация с 0)
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
     * Значение начала строки и количества
     * строк для каждого процесса
     */
    for (int i = 0; i < ProcNum; i++){
        int rem = size % ProcNum; //остаток
        int quo = size / ProcNum;
        int NumOfLines = 0;
        int startLineProc = 0;
        if (i < rem) {
            NumOfLines = quo + 1;
            startLineProc = (i) * (NumOfLines);

        } else {
            NumOfLines = quo;
            startLineProc = (rem * (NumOfLines + 1)) + ((i - rem) * NumOfLines);
        }
        countRowsAtProc[i] = NumOfLines;
        shift[i] = startLineProc;
    }

    u = new double [NumberOfLines];
    for(int i = 0; i < NumberOfLines; i++) {
        u[i] = cos(2 * M_PI * i / size);
    }

    std::cout <<"1";
    b = new double [NumberOfLines];
    matrixAndVectorMul(A, u, b, size, shift, countRowsAtProc, ProcNum, ProcRank);

    x = new double [NumberOfLines];
    r = new double [NumberOfLines];
    z = new double [NumberOfLines];
    Ax = new double [NumberOfLines];
    Az = new double [NumberOfLines];
    std::fill(x, x + countRowsAtProc[ProcRank], 0.0f);
    std::fill(r, r + countRowsAtProc[ProcRank], 0.0f);
    std::fill(z, z + countRowsAtProc[ProcRank], 0.0f);
    std::fill(Ax, Ax + countRowsAtProc[ProcRank], 0.0f);
    std::fill(Az, Az + countRowsAtProc[ProcRank], 0.0f);
}

int main(int argc, char *argv[]){

    int N = 10;

    MPI_Init(&argc, &argv);
    int ProcNum, ProcRank;
    MPI_Comm_size (MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank (MPI_COMM_WORLD, &ProcRank);

    double *A;
    double *b;
    double *x;
    double *u;
    double *r;
    double *z;
    double *Ax;
    double *Az;

    int *countRowsAtProc;
    int *shift;

    init(x, A, b, u, r, z, Ax, Az, N, ProcRank, ProcNum, shift, countRowsAtProc);


    double  alpha,
            beta,
            firstScalar,
            secondScalar;


    double startTime = MPI_Wtime();

    /**
     * Далее у нас идёт чисто метод сопряжённых градиентов
     * r_0 = b - Ax_0
    */
    matrixAndVectorMul(A, x, Ax, N, shift, countRowsAtProc, ProcNum, ProcRank);
    vectorsSub(b, Ax, r, countRowsAtProc[ProcRank]);

    /**
     * z_0 = r_0
    */
    std::copy(r, r + countRowsAtProc[ProcRank], z);

    /**
     * Выполняем итерации в цикле до тех пор, пока критерий
     * завершения счёта ||r_n|| / ||b|| не будет меньше заданной точности, именно
     * того эпсилон, который мы задаём сами, чем меньше - тем лучше
    */
    while (epsilon <= finishCount(r, b, countRowsAtProc[ProcRank])) {
        MPI_Barrier(MPI_COMM_WORLD);
        /**
         * Скалярное произведение(r_n, r_n)
        */
        firstScalar = scalarVectorsMultiplication(r, r, countRowsAtProc[ProcRank]);

        /**
        * Скалярное произведение(Az_n, z_n)
        */
        matrixAndVectorMul(A, z, Az, N, shift, countRowsAtProc, ProcNum, ProcRank);

        secondScalar = scalarVectorsMultiplication(Az, z, countRowsAtProc[ProcRank]);

        /**
         * alpha =(r_n, r_n)/(Az_n, z_n)
        */
        alpha = firstScalar / secondScalar;

        /**
         * x_n+1 = x_n + alpha_n+1*z_n
         * r_n+1 = r_n - alpha_n+1*A*z_n
        */
        for (int i = 0; i < countRowsAtProc[ProcRank]; i++) {
            x[i] += alpha * z[i];
            r[i] -= alpha * Az[i];
        }

        secondScalar = scalarVectorsMultiplication(r, r, countRowsAtProc[ProcRank]);

        /**
         * beta =(r_n+1, r_n+1)/(r_n, r_n)
        */
        beta = secondScalar / firstScalar;

        /**
         * z_n+1 = r_n+1 + beta_n+1*z_n
        */
        for (int i = 0; i < countRowsAtProc[ProcRank]; i++) {
            z[i] = r[i] + (beta * z[i]);
        }
    }
    double endTime = MPI_Wtime();
    double duration = endTime - startTime;

    /**
     * Сравниваем наш полученный вектор x и наш эталонный вектор u
     * Ура, они одинаковые
     */
    std::cout << "Compare u[] and x[] is equals" << std::endl;
    std::cout << "Time: " << duration << " sec" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < countRowsAtProc[ProcRank]; i++) {
        std::cout << "index :" << i << " res: " << x[i] << " main: " << u[i] << std::endl;
    }


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

/*int *pReceiveNum;  // Количество элементов, посылаемых процессом
    int *pReceiveInd;  // Индекс элемента данных в результирующем
    // векторе
    int RestRows = size; // Количество строк матрицы, которые еще не
    // распределены

    // Выделение памяти для временных объектов
    pReceiveNum = new int [ProcNum];
    pReceiveInd = new int [ProcNum];

    // Определение положения блоков результирующего вектора
    pReceiveInd[0] = 0;
    pReceiveNum[0] = size/ProcNum;
    for (int i = 1; i < ProcNum; i++) {
        RestRows -= pReceiveNum[i-1];
        pReceiveNum[i] = RestRows / (ProcNum - i);
        pReceiveInd[i] = pReceiveInd[i - 1] + pReceiveNum[i - 1];
    }
    // Сбор всего результирующего вектора на всех процессах
    MPI_Allgatherv(resultVector, countRowsAtProc[ProcRank],
                   MPI_DOUBLE, resultVector, pReceiveNum, pReceiveInd,
                   MPI_DOUBLE, MPI_COMM_WORLD);

    // Освобождение памяти
    delete [] pReceiveNum;
    delete [] pReceiveInd;*/


/*///считаем A * vector - записываем в буфер
auto * mulBuf = new double [countRowsAtProc * size]; //буффер для умножения
for (int k = 0; k < countRowsAtProc * size; ++k)
    mulBuf[k] = 0.0f;

for (int i = 0; i < countRowsAtProc; ++i) {
    resultVector[i] = 0.0f;
    for (int j = 0; j < size; ++j) {
        resultVector[i] += matrix[i * size + j] * vector1[i];
        mulBuf[i * size + j] += matrix[i * size + j] * vector1[i];
    }
}
*//** построчно суммируем элементы буфера
     *  так как при инициализации матрицу транспонировал, чтобы вычилсения были быстрее
     *//*
    auto * buffer = new double [size];
    for (int k = 0; k < size; ++k)
        buffer[k] = 0.0f;

    for (int i = 0; i < countRowsAtProc; ++i) {
        for (int j = 0; j < size; ++j) {
            buffer[j] += mulBuf[i * size + j];
        }
    }
    *//** теперь нужно проссумировать соответсвующии блоки в каждом процессе
     *  каждый процесс отправляет нужную часть своего просуммированого столбца
     *  (с shift[i] по countRowsAtProc[i] процессу i)
     *  процесс с номером i принимает эти данные и суммирует в соотвестующих яейках A
     *//*
    MPI_Allreduce(MPI_IN_PLACE, buffer, size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for(int i = 0; i < countRowsAtProc; i++){
        resultVector[i] = buffer[i + shift];
    }

    delete [] buffer;
    delete [] mulBuf;*/