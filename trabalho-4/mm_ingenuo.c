#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> // para medir o tempo de execução com 'omp_get_wtime()'

long n = -1;
double* mat1 = NULL;
double* mat2 = NULL;
double* prod = NULL;

unsigned int seed = 2023;

void tratar_args(int argc, char* argv[]);
void iniciar();
void exportar_bin();
void finalizar();

void multiplicar_mat() {
    for (long i = 0; i < n; ++i)
        for (long j = 0; j < n; ++j)
            for (long k = 0; k < n; ++k)
                prod[i*n + j] += mat1[i*n + k] * mat2[k*n + j];
}


int main(int argc, char* argv[]) {
    tratar_args(argc, argv);
    iniciar();

    double t_inicio = omp_get_wtime();

    multiplicar_mat();

    double t_fim = omp_get_wtime();
    printf("%ld x %ld em %.15lf segundos\n", n, n, t_fim - t_inicio);

    // exportar_bin();
    finalizar();
    return 0;
}


void tratar_args(int argc, char* argv[]) {
    if (argc == 2) {
        n = atol(argv[1]);
    } else {
        fprintf(stderr, "Número de argumentos inválido!\n");
        exit(1);
    }
}

void iniciar() {
    size_t size = n*n * sizeof(double);
    mat1 = (double*) malloc(size);
    mat2 = (double*) malloc(size);
    prod = (double*) malloc(size);
    if (mat1 == NULL || mat2 == NULL || prod == NULL) {
        fprintf(stderr, "Não foi possível alocar memória!\n");
        exit(2);
    }

    // gera valores de mat1 e mat2
    srand(seed);
    double interv_inicio = -100.0;
    double interv_fim = 100.0;
    for (long i = 0; i < n; ++i)
        for (long j = 0; j < n; ++j) {
            mat1[i*n + j] = (((double) rand() / RAND_MAX) * (interv_fim - interv_inicio)) + interv_inicio;
            mat2[i*n + j] = (((double) rand() / RAND_MAX) * (interv_fim - interv_inicio)) + interv_inicio;
        }

    // zera todos os valores de prod
    for (long i = 0; i < n; ++i)
        for (long j = 0; j < n; ++j)
            prod[i*n + j] = 0.0;
}

void exportar_bin() {
    FILE* arquivo;
    arquivo = fopen("mat_prod", "wb");
    if (arquivo == NULL) {
        fprintf(stderr, "Não foi possível criar o arquivo!\n");
        exit(3);
    }

    // dimensão
    fwrite(&n, sizeof(long), 1, arquivo);
    // elementos
    fwrite(prod, sizeof(double), (n * n), arquivo);

    fclose(arquivo);
}

void finalizar() {
    free(mat1);
    free(mat2);
    free(prod);
}
