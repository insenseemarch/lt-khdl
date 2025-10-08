#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double** allocate_matrix(int n)
{
    double** matrix = (double**)malloc(n*sizeof(double*));
    for(int i=0; i<n; ++i)
    {
        matrix[i] = (double*)malloc(n*sizeof(double));
    }
    return matrix;
}

void free_matrix(double** matrix, int n)
{
    for(int i=0; i<n; ++i)
    {
        free(matrix[i]);
    }
    free(matrix);
}

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

int main(int argc, char *argv[])
{
    int n= atoi(argv[1]);
    if(n <= 0)
    {
        fprintf(stderr, "Kích thước ma trận phải là số nguyên dương!");
        return 1;
    }
    srand(time(NULL));

    double** A = allocate_matrix(n);
    double** B = allocate_matrix(n);
    double** C = allocate_matrix(n);

    for(int i=0;i<n;++i)
    {
        for(int j=0; j<n; ++j)
        {
            A[i][j] = (float)rand() / RAND_MAX;
            B[i][j] = (float)rand() / RAND_MAX;
        }
    }

    printf("Dang thuc hien nhan hai ma tran co kich thuoc %d x %d...\n", n, n);

    clock_t time = clock();

    matrix_multiplication(A, B, C, n);

    time = clock() - time;

    double execution_time_ms = ((double)(time) / CLOCKS_PER_SEC) * 1000.0;
    printf("Thoi gian thuc thi: %.2f milliseconds\n", execution_time_ms);

    free_matrix(A, n);
    free_matrix(B, n);
    free_matrix(C, n);

    return 0;
}
