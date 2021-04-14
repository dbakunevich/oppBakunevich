#include <iostream>
#include <cmath>
#include <ctime>
#include <cstring>
#include <omp.h>

#define epsilon 10e-7

void matrixAndVectorMul(const double *matrix, const double *vector1, double *resultVector, int size) {
    int i, j;
    #pragma omp parallel for shared(matrix, vector1, resultVector, size) default (none)  private (i, j)
    for (i = 0; i < size; i++) {
        resultVector[i] = 0.0f;
        for (j = 0; j < size; j++) {
            resultVector[i] += matrix[i * size + j] * vector1[j];
        }
    }
}

void init(double *&x, double *&A, double *&b, double *&u, int size){
    int i, j;
    A = new double [size * size];
    #pragma omp parallel for shared(A, size) default (none) private (i, j)
    for (i = 0; i < size; i++) {
        for(j = 0; j < size; j++) {
            A[i * size + j] = 1.0f;
            if (i == j){
                A[i * size + j] = 1.0f * size;
            }
        }
    }
    u = new double [size];
    #pragma omp parallel for shared(u, size) default (none)  private (i)
    for(i = 0; i < size; i++) {
        u[i] = cos(2 * M_PI * i / size);
    }

    b = new double [size];
    matrixAndVectorMul(A, u, b, size);

    x = new double [size];
    #pragma omp parallel for shared(x, size) default (none)  private (i)
    for(i = 0; i < size; i++) {
        x[i] = 0.0f;
    }

}

int main(int argc, char *argv[]) {
    int size = 30000;

    double *A = nullptr;
    double *b = nullptr;
    double *x = nullptr;
    double *u = nullptr;

    init(x, A, b, u, size);

    auto *r = new double [size];
    auto *z = new double [size];
    auto *Ax = new double [size];
    auto *Az = new double [size];
    std::fill(r, r+size, 0);
    std::fill(z, z+size, 0);
    std::fill(Ax, Ax+size, 0);
    std::fill(Az, Az+size, 0);

    double  alpha = 0,
            beta = 0,
            firstScalar = 0,
            secondScalar = 0,
            lenOfVec_r = 0,
            lenOfVec_b = 0;
    int i, j;
    bool flag = true;
    long double exit = 0;


    struct timespec start{}, finish{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    /**
    * Далее у нас идёт чисто метод сопряжённых градиентов
    * r_0 = b - Ax_0
    */
    #pragma omp parallel for shared(r, b, Ax, size) private (i) default (none)
    for (i = 0; i < size; i++) {
        r[i] = b[i] - Ax[i];
    }

    /**
    * z_0 = r_0
    */
    std::copy(r, r + size, z);


    /**
    * Выполняем итерации в цикле до тех пор, пока критерий
    * завершения счёта ||r_n|| / ||b|| не будет меньше заданной точности, именно
    * того эпсилон, который мы задаём сами, чем меньше - тем лучше
    */
    #pragma omp parallel private(i, j)
    while (flag)
    {
        /**
        * Скалярное произведение(Az_n, z_n)
        */
        #pragma omp for
        for (i = 0; i < size; i++) {
            Az[i] = 0.0f;
            #pragma omp barier
            for (j = 0; j < size; j++) {
                Az[i] += A[i * size + j] * z[j];
            }
        }
        /**
         * Скалярное произведение(r_n, r_n)
        */
        #pragma omp for reduction(+:firstScalar)
        for (i = 0; i < size; i++) {
            firstScalar += r[i] * r[i];
        }



        #pragma omp for reduction(+:secondScalar)
        for (i = 0; i < size; i++) {
            secondScalar += Az[i] * z[i];
        }

        /**
        * alpha =(r_n, r_n)/(Az_n, z_n)
        */

        #pragma omp single
        alpha = firstScalar / secondScalar;

        /**
        * x_n+1 = x_n + alpha_n+1*z_n
        * r_n+1 = r_n - alpha_n+1*A*z_n
        */

        #pragma omp for
        for (i = 0; i < size; i++)
        {
            x[i] += alpha * z[i];
            r[i] -= alpha * Az[i];
        }

        #pragma omp for reduction(+:secondScalar)
        for (i = 0; i < size; i++) {
            secondScalar += r[i] * r[i];
        }

        #pragma omp single
        {
            lenOfVec_r = 0;
            lenOfVec_b = 0;
        }
        #pragma omp for reduction(+:lenOfVec_r)
        for (i = 0; i < size; i++) {
            lenOfVec_r += pow(r[i], 2);
        }
        #pragma omp for reduction(+:lenOfVec_b)
        for (i = 0; i < size; i++) {
            lenOfVec_b += pow(b[i], 2);
        }
        #pragma omp single
        {
        exit = sqrt(lenOfVec_r) / sqrt(lenOfVec_b);
        if (epsilon > exit)
            flag = !flag;
            /**
            * beta =(r_n+1, r_n+1)/(r_n, r_n)
            */
            beta = secondScalar / firstScalar;
        }

        /**
        * z_n+1 = r_n+1 + beta_n+1*z_n
        */
        #pragma omp for
        for (i = 0; i < size; i++)
        {
            z[i] = r[i] + (beta *z[i]);
        }

    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &finish);

    /**
     * Сравниваем наш полученный вектор x и наш эталонный вектор u
     * Ура, они одинаковые
     */
    std::cout << "Compare u[] and x[] is equals" << std::endl;

    for (i = 0; i < size; i++) {
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