#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> // para medir o tempo de execução com 'omp_get_wtime()'

// gcc -DL1D_CACHE_TAM=$(getconf LEVEL1_DCACHE_LINESIZE)
#define BLOCO_TAM (L1D_CACHE_TAM / sizeof(double))

long n;
double* mat1 = NULL;
double* mat2 = NULL;
double* prod = NULL;

unsigned int seed = 2023;

void tratar_args(int argc, char* argv[]);
void iniciar();
void exportar_bin();
void finalizar();

void multiplicar_mat() {    
    for (long i = 0; i < n; i += BLOCO_TAM) {
        for (long j = 0; j < n; j += BLOCO_TAM) {
            for (long k = 0; k < n; k += BLOCO_TAM) {
                double* p_mat1 = mat1 + i*n + k;
                double* p_prod = prod + i*n + j;
                for (int i2 = 0; i2 < BLOCO_TAM; ++i2) {
                    double* p_mat2 = mat2 + k*n + j;
                    for (int k2 = 0; k2 < BLOCO_TAM; ++k2) {                       
                        for (int j2 = 0; j2 < BLOCO_TAM; ++j2)
                            p_prod[j2] += p_mat1[k2] * p_mat2[j2];
                        p_mat2 += n;
                    }
                    p_mat1 += n;
                    p_prod += n;
                }
            }
        }
    }
}


int main(int argc, char* argv[]) {
    tratar_args(argc, argv);
    iniciar();

    double t_inicio = omp_get_wtime();

    multiplicar_mat();

    double t_fim = omp_get_wtime();
    printf("Tempo de execução: %.15lf\n", t_fim - t_inicio);

    exportar_bin();
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

    printf("n = %ld\n", n);
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
