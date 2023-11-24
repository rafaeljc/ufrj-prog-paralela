#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> // para medir o tempo de execução com 'omp_get_wtime()'

// gcc -DL1D_CACHE_TAM=$(getconf LEVEL1_DCACHE_LINESIZE)
#define BLOCO_TAM (L1D_CACHE_TAM / sizeof(double))

long n = -1;
double* mat1 = NULL;
double* mat2 = NULL;
double* mat2_t = NULL;
double* prod = NULL;

unsigned int seed = 2023;

void tratar_args(int argc, char* argv[]);
void alocar_mem(double** mat, size_t size);
void iniciar();
void transpor_mat(double* mat, double* mat_t);
void exportar_bin();
void finalizar();

void multiplicar_mat() {
    transpor_mat(mat2, mat2_t);
    
    long num = n / BLOCO_TAM;
    for (long i = 0; i < num; ++i) {
        for (long j = 0; j < num; ++j) {
            for (int k = 0; k < BLOCO_TAM; ++k) {
                double* p_prod = prod + i*BLOCO_TAM*n + j*BLOCO_TAM + k*n;
                for (int m = 0; m < BLOCO_TAM; ++m) {
                    double soma = 0.0;
                    for (long r = 0; r < num; ++r) {
                        double* p_mat1 = mat1 + i*BLOCO_TAM*n + r*BLOCO_TAM + k*n;
                        double* p_mat2_t = mat2_t + j*BLOCO_TAM*n + r*BLOCO_TAM + m*n;
                        for (int p = 0; p < BLOCO_TAM; ++p) {
                            soma += (*p_mat1) * (*p_mat2_t);
                            p_mat1++;
                            p_mat2_t++;
                        }
                    }
                    *p_prod = soma;
                    p_prod++;
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

void alocar_mem(double** mat, size_t size) {
    if (posix_memalign((void**) mat, L1D_CACHE_TAM, size)) {
        fprintf(stderr, "Não foi possível alocar memória!\n");
        exit(2);
    }
}

void iniciar() {
    size_t size = n*n * sizeof(double);
    alocar_mem(&mat1, size);
    alocar_mem(&mat2, size);
    alocar_mem(&mat2_t, size);
    alocar_mem(&prod, size);

    // gera valores de mat1 e mat2
    srand(seed);
    double interv_inicio = -100.0;
    double interv_fim = 100.0;
    for (long i = 0; i < n; ++i)
        for (long j = 0; j < n; ++j) {
            mat1[i*n + j] = (((double) rand() / RAND_MAX) * (interv_fim - interv_inicio)) + interv_inicio;
            mat2[i*n + j] = (((double) rand() / RAND_MAX) * (interv_fim - interv_inicio)) + interv_inicio;
        }
}

void transpor_mat(double* mat, double* mat_t) {
    for (long i = 0; i < n; ++i) {
        double* p_mat = mat + i;
        double* p_mat_t = mat_t + i*n;        
        for (long j = 0; j < n; ++j) {
            *p_mat_t = *p_mat;            
            p_mat += n;
            p_mat_t++;
        }
    }        
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
    free(mat2_t);
    free(prod);
}
