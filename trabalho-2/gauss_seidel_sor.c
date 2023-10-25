#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Gauss-Seidel SOR (successive over-relaxation)
#define NUM_ITER 4098
#define W 0.5  // fator de relaxamento

// problema
#define N 1022
#define TEMP_INICIAL 20.0
#define TEMP_BORDA 20.0
#define TEMP_FONTE_CALOR 100.0
#define I_FONTE_CALOR 800
#define J_FONTE_CALOR 800

void iniciar();
double gauss_seidel_sor(int i, int j);

double* a = NULL;
double* a_anterior = NULL;


int main(int argc, char* argv[]) {
    iniciar();

    int k = 0;    
    while (k < NUM_ITER) {
        // fase red
        for (int i = 1; i < N - 1; i += 2)
            for (int j = 1; j < N - 1; j += 2)
                if (i != I_FONTE_CALOR || j != J_FONTE_CALOR)
                    a[i*N + j] = gauss_seidel_sor(i, j);  
        for (int i = 2; i < N - 1; i += 2)
            for (int j = 2; j < N - 1; j += 2)
                if (i != I_FONTE_CALOR || j != J_FONTE_CALOR)
                    a[i*N + j] = gauss_seidel_sor(i, j);
 
        // fase black
        for (int i = 1; i < N - 1; i += 2)
            for (int j = 2; j < N - 1; j += 2)
                if (i != I_FONTE_CALOR || j != J_FONTE_CALOR)
                    a[i*N + j] = gauss_seidel_sor(i, j);                  
        for (int i = 2; i < N - 1; i += 2)
            for (int j = 1; j < N - 1; j += 2)
                if (i != I_FONTE_CALOR || j != J_FONTE_CALOR) 
                    a[i*N + j] = gauss_seidel_sor(i, j);                  

        double* tmp = a;
        a = a_anterior;
        a_anterior = tmp;

        ++k;
    }

    double erro = 0.0;
    for (int i = 1; i < N - 1; ++i)
        for (int j = 1; j < N - 1; ++j)
            erro = fmax(fabs(a[i*N + j] - a_anterior[i*N + j]), erro);
    
    printf("Após %d iterações, o erro era: %.15lf\n", k, erro);

    free(a);
    free(a_anterior);

    return 0;
}


void iniciar() {
    size_t size = N * N * sizeof(double);
    a = (double*) malloc(size);
    a_anterior = (double*) malloc(size);
    if (!a || !a_anterior) {
        fprintf(stderr, "Não foi possível alocar memória!\n");
        exit(1);
    }

    // bordas laterais
    for (int i = 1; i < N - 1; ++i) {
        a_anterior[i*N] = TEMP_BORDA;
        a_anterior[i*N + N - 1] = TEMP_BORDA; // i*N + j
    }

    // borda superior e inferior
    for (int j = 0; j < N; ++j) {
        a_anterior[j] = TEMP_BORDA;
        a_anterior[(N - 1)*N + j] = TEMP_BORDA;  // i*N + j
    }

    // valores internos
    for (int i = 1; i < N - 1; ++i)
        for (int j = 1; j < N - 1; ++j)
            a_anterior[i*N + j] = TEMP_INICIAL;

    // fonte de calor
    a_anterior[I_FONTE_CALOR*N + J_FONTE_CALOR] = TEMP_FONTE_CALOR;

    memcpy(a, a_anterior, size);
}

double gauss_seidel_sor(int i, int j) {
    double gauss_seidel = 0.25 * (a_anterior[(i + 1)*N + j] + a[(i - 1)*N + j] + a_anterior[i*N + j + 1] + a[i*N + j - 1]);
    // 1 multiplicação a menos que W*gauss_seidel + (1.0 - W)*a_anterior[i*N + j]
    return a_anterior[i*N + j] + W*(gauss_seidel - a_anterior[i*N + j]);
}
