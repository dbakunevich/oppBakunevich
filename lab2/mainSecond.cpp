#include <iostream>
#include <cmath>
#include <ctime>
#include <omp.h>

#define epsilon 10e-7

double scalarVectorsMultiplication(const double * vector1, const double * vector2, int size) {
    double resultScalar = 0;
    for (int i = 0; i < size; i++) {
        resultScalar += vector1[i] * vector2[i];
    }
    return resultScalar;
}

void vectorsSub(const double *vector1, const double *vector2, double *resultVector, int size) {
    for (int i = 0; i < size; i++) {
        resultVector[i] = vector1[i] - vector2[i];
    }
}

void matrixAndVectorMul(const double *matrix, const double *vector1, double *resultVector, int size) {
    int i, j;
    #pragma omp parallel for shared(matrix, vector1, resultVector) private (i, j)
    for (i = 0; i < size; i++) {
        resultVector[i] = 0.0f;
        for (j = 0; j < size; j++) {
            resultVector[i] += matrix[i * size + j] * vector1[j];
        }
    }
}

void init(double *&x, double *&A, double *&b, double *&u, int size){

    A = new double [size * size];
    for (int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            A[i * size + j] = 1.0f;
            if (i == j){
                A[i * size + j] = 2.0f;
            }
        }
    }
    u = new double [size];
    for(int i = 0; i < size; i++) {
        u[i] = cos(2 * M_PI * i / size);
    }

    b = new double [size];
    matrixAndVectorMul(A, u, b, size);

    x = new double [size];
    for(int i = 0; i < size; i++) {
        x[i] = 0.0f;
    }

}

double finishCount(double *rVector, double *bVector, int size) {
    double result;
    double lenOfVec_r = 0;
    double lenOfVec_b = 0;
    for (int i = 0; i < size; i++) {
        lenOfVec_r += pow(rVector[i], 2);
    }
    for (int i = 0; i < size; i++) {
        lenOfVec_b += pow(bVector[i], 2);
    }
    result = sqrt(lenOfVec_r) / sqrt(lenOfVec_b);
    return result;
}

int main(int argc, char *argv[]) {
    int size = 20000;


    double *A = nullptr;
    double *b = nullptr;
    double *x = nullptr;
    double *u = nullptr;


    init(x, A, b, u, size);

    auto *r = new double [size];
    auto *z = new double [size];
    auto *Ax = new double [size];
    auto *Az = new double [size];

    double  alpha,
            beta,
            firstScalar,
            secondScalar;

    struct timespec start{}, finish{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    /**
     * Далее у нас идёт чисто метод сопряжённых градиентов
     * r_0 = b - Ax_0
    */
    vectorsSub(b, Ax, r, size);

    /**
     * z_0 = r_0
    */
    std::copy(r, r + size, z);


    /**
     * Выполняем итерации в цикле до тех пор, пока критерий
     * завершения счёта ||r_n|| / ||b|| не будет меньше заданной точности, именно
     * того эпсилон, который мы задаём сами, чем меньше - тем лучше
    */
    while (epsilon <= finishCount(r, b, size)) {
        /**
         * Скалярное произведение(r_n, r_n)
        */
        firstScalar = scalarVectorsMultiplication(r, r, size);

        /**
        * Скалярное произведение(Az_n, z_n)
        */
        matrixAndVectorMul(A, z, Az, size);
        secondScalar = scalarVectorsMultiplication(Az, z, size);

        /**
         * alpha =(r_n, r_n)/(Az_n, z_n)
        */
        alpha = firstScalar / secondScalar;

        /**
         * x_n+1 = x_n + alpha_n+1*z_n
         * r_n+1 = r_n - alpha_n+1*A*z_n
        */
        for (int i = 0; i < size; i++) {
            x[i] += alpha * z[i];
            r[i] -= alpha * Az[i];
        }
        secondScalar = scalarVectorsMultiplication(r, r, size);

        /**
         * beta =(r_n+1, r_n+1)/(r_n, r_n)
        */
        beta = secondScalar / firstScalar;

        /**
         * z_n+1 = r_n+1 + beta_n+1*z_n
        */
        for (int i = 0; i < size; i++) {
            z[i] = r[i] + (beta * z[i]);
        }
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &finish);

    /**
     * Сравниваем наш полученный вектор x и наш эталонный вектор u
     * Ура, они одинаковые
     */
    std::cout << "Compare u[] and x[] is equals" << std::endl;

    for (int i = 0; i < size; i++) {
        std::cout << "index :" << i << " res: " << x[i] << " main: " << u[i] << std::endl;
    }

    std::cout << "Time: " << ((double) finish.tv_sec - start.tv_sec + 0.000000001 * (double) (finish.tv_nsec - start.tv_nsec)) << '\n';

    delete[] A;
    delete[] b;
    delete[] x;
    delete[] u;
    delete[] r;
    delete[] z;
    delete[] Ax;
    delete[] Az;

    return 0;
}