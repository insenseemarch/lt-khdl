#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Cấp phát động ma trận
double** allocate_matrix(int n)
{
    double** matrix = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; ++i)
    {
        matrix[i] = (double*)malloc(n * sizeof(double));
    }
    return matrix;
}

// Giải phóng vùng nhớ
void free_matrix(double** matrix, int n)
{
    for (int i = 0; i < n; ++i)
    {
        free(matrix[i]);
    }
    free(matrix);
}

// Thuật toán nhân hai ma trận
void matrix_multiplication(double** A, double** B, double** C, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            C[i][j] = 0.0;
            for (int k = 0; k < n; ++k)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int n = atoi(argv[1]);
    if (n <= 0) return 1;

    srand(time(NULL));
    double** A = allocate_matrix(n);
    double** B = allocate_matrix(n);
    double** C = allocate_matrix(n);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            A[i][j] = (double)rand() / RAND_MAX;
            B[i][j] = (double)rand() / RAND_MAX;
        }
    
    // Bắt đầu đếm thời gian
    clock_t t = clock();
    matrix_multiplication(A, B, C, n);

    // Kết thúc đếm thời gian
    t = clock() - t;

    double execution_time_ms = ((double)t / CLOCKS_PER_SEC) * 1000.0;
    printf("%.6f\n", execution_time_ms); 

    free_matrix(A, n);
    free_matrix(B, n);
    free_matrix(C, n);
    return 0;
}