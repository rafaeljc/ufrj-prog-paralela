#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// gcc -DL1D_CACHE_TAM=$(getconf LEVEL1_DCACHE_LINESIZE)
#define BLOCO_TAM (L1D_CACHE_TAM / sizeof(double))
#define STRASSEN_LIM 1000

long n;
double* mat1 = NULL;
double* mat2 = NULL;
double* prod = NULL;

unsigned int seed = 2023;

void tratar_args(int argc, char* argv[]);
void alocar_mem(double** mat, size_t size);
void iniciar();
void exportar_bin();
void finalizar();
void strassen(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, 
              double* _prod, long _prod_jmp);
void mm_otimizado(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, 
                  double* _prod, long _prod_jmp);
void transpor_mat(long _n, double* _mat, long _mat_jmp, double* _mat_t, long _mat_t_jmp);
void somar_mat(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, 
               double* _adic, long _adic_jmp);
void subtrair_mat(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, 
                  double* _subt, long _subt_jmp);


int main(int argc, char* argv[]) {
    tratar_args(argc, argv);
    iniciar();

    double t_inicio = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            strassen(n, (double*) mat1, n, (double*) mat2, n, (double*) prod, n);
        }
    }

    double t_fim = omp_get_wtime();
    printf("Tempo de execução: %.15lf\n", t_fim - t_inicio);

    exportar_bin();
    finalizar();
    return 0;
}


void strassen(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, 
              double* _prod, long _prod_jmp) {
    if (_n <= STRASSEN_LIM) {
        mm_otimizado(_n, _mat1, _mat1_jmp, _mat2, _mat2_jmp, _prod, _prod_jmp);
        return;
    }

    long novo_n = _n / 2;
    size_t novo_size = novo_n*novo_n * sizeof(double);

    double* mat1_00 = _mat1;
    double* mat1_01 = _mat1 + novo_n;
    double* mat1_10 = _mat1 + novo_n*_mat1_jmp;
    double* mat1_11 = _mat1 + novo_n*_mat1_jmp + novo_n;

    double* mat2_00 = _mat2;
    double* mat2_01 = _mat2 + novo_n;
    double* mat2_10 = _mat2 + novo_n*_mat2_jmp;
    double* mat2_11 = _mat2 + novo_n*_mat2_jmp + novo_n;

    double* prod_00 = _prod;
    double* prod_01 = _prod + novo_n;
    double* prod_10 = _prod + novo_n*_prod_jmp;
    double* prod_11 = _prod + novo_n*_prod_jmp + novo_n;

    double* s0 = NULL;
    double* s1 = NULL;
    double* s2 = NULL;
    double* s3 = NULL;
    double* s4 = NULL;
    double* s5 = NULL;
    double* s6 = NULL;
    double* s7 = NULL;
    double* s8 = NULL;
    double* s9 = NULL;

    double* p0 = NULL;
    double* p1 = NULL;
    double* p2 = NULL;
    double* p3 = NULL;
    double* p4 = NULL;
    double* p5 = NULL;
    double* p6 = NULL;

    #pragma omp taskgroup
    {
        // S0 = B01 - B11
        // P0 = A00 * S0
        #pragma omp task depend(out: p0) default(none) firstprivate(s0, novo_size, novo_n, mat2_01, mat2_11, _mat2_jmp, mat1_00, _mat1_jmp) shared(p0)
        {
            alocar_mem(&s0, novo_size);
            alocar_mem(&p0, novo_size);
            subtrair_mat(novo_n, mat2_01, _mat2_jmp, mat2_11, _mat2_jmp, s0, novo_n);    
            strassen(novo_n, mat1_00, _mat1_jmp, s0, novo_n, p0, novo_n);
            free(s0);
        }        

        // S1 = A00 + A01
        // P1 = S1 * B11
        #pragma omp task depend(out: p1) default(none) firstprivate(s1, novo_size, novo_n, mat1_00, mat1_01, _mat1_jmp, mat2_11, _mat2_jmp) shared(p1)
        {
            alocar_mem(&s1, novo_size);
            alocar_mem(&p1, novo_size);
            somar_mat(novo_n, mat1_00, _mat1_jmp, mat1_01, _mat1_jmp, s1, novo_n);    
            strassen(novo_n, s1, novo_n, mat2_11, _mat2_jmp, p1, novo_n);
            free(s1);
        }        

        // S2 = A10 + A11
        // P2 = S2 * B00
        #pragma omp task depend(out: p2) default(none) firstprivate(s2, novo_size, novo_n, mat1_10, mat1_11, _mat1_jmp, mat2_00, _mat2_jmp) shared(p2)
        {
            alocar_mem(&s2, novo_size);
            alocar_mem(&p2, novo_size);
            somar_mat(novo_n, mat1_10, _mat1_jmp, mat1_11, _mat1_jmp, s2, novo_n);    
            strassen(novo_n, s2, novo_n, mat2_00, _mat2_jmp, p2, novo_n);
            free(s2);
        }

        // S3 = B10 - B00
        // P3 = A11 * S3
        #pragma omp task depend(out: p3) default(none) firstprivate(s3, novo_size, novo_n, mat2_10, mat2_00, _mat2_jmp, mat1_11, _mat1_jmp) shared(p3)
        {
            alocar_mem(&s3, novo_size);
            alocar_mem(&p3, novo_size);
            subtrair_mat(novo_n, mat2_10, _mat2_jmp, mat2_00, _mat2_jmp, s3, novo_n);    
            strassen(novo_n, mat1_11, _mat1_jmp, s3, novo_n, p3, novo_n);
            free(s3);
        }        

        // S4 = A00 + A11
        // S5 = B00 + B11
        // P4 = S4 * S5
        #pragma omp task depend(out: p4) default(none) firstprivate(s4, s5, novo_size, novo_n, mat1_00, mat1_11, _mat1_jmp, mat2_00, mat2_11, _mat2_jmp) shared(p4)
        {
            alocar_mem(&s4, novo_size);
            alocar_mem(&s5, novo_size);
            alocar_mem(&p4, novo_size);
            somar_mat(novo_n, mat1_00, _mat1_jmp, mat1_11, _mat1_jmp, s4, novo_n); 
            somar_mat(novo_n, mat2_00, _mat2_jmp, mat2_11, _mat2_jmp, s5, novo_n);   
            strassen(novo_n, s4, novo_n, s5, novo_n, p4, novo_n);
            free(s4);
            free(s5);
        }        

        // S6 = A01 - A11
        // S7 = B10 + B11
        // P5 = S6 * S7
        #pragma omp task depend(out: p5) default(none) firstprivate(s6, s7, novo_size, novo_n, mat1_01, mat1_11, _mat1_jmp, mat2_10, mat2_11, _mat2_jmp) shared(p5)
        {
            alocar_mem(&s6, novo_size);
            alocar_mem(&s7, novo_size);
            alocar_mem(&p5, novo_size);
            subtrair_mat(novo_n, mat1_01, _mat1_jmp, mat1_11, _mat1_jmp, s6, novo_n); 
            somar_mat(novo_n, mat2_10, _mat2_jmp, mat2_11, _mat2_jmp, s7, novo_n);   
            strassen(novo_n, s6, novo_n, s7, novo_n, p5, novo_n);
            free(s6);
            free(s7);
        }        

        // S8 = A00 - A10
        // S9 = B00 + B01
        // P6 = S8 * S9
        #pragma omp task depend(out: p6) default(none) firstprivate(s8, s9, novo_size, novo_n, mat1_00, mat1_10, _mat1_jmp, mat2_00, mat2_01, _mat2_jmp) shared(p6)
        {
            alocar_mem(&s8, novo_size);
            alocar_mem(&s9, novo_size);
            alocar_mem(&p6, novo_size);
            subtrair_mat(novo_n, mat1_00, _mat1_jmp, mat1_10, _mat1_jmp, s8, novo_n); 
            somar_mat(novo_n, mat2_00, _mat2_jmp, mat2_01, _mat2_jmp, s9, novo_n);   
            strassen(novo_n, s8, novo_n, s9, novo_n, p6, novo_n);
            free(s8);
            free(s9);
        }        

        // C00 = P4 + P3 - P1 + P5
        #pragma omp task depend(in: p4, p3, p1, p5) default(none) firstprivate(novo_n, prod_00, _prod_jmp) shared(p4, p3, p1, p5)
        {
            somar_mat(novo_n, p4, novo_n, p3, novo_n, prod_00, _prod_jmp);
            subtrair_mat(novo_n, prod_00, _prod_jmp, p1, novo_n, prod_00, _prod_jmp);
            somar_mat(novo_n, prod_00, _prod_jmp, p5, novo_n, prod_00, _prod_jmp);
        }

        // C01 = P0 + P1
        #pragma omp task depend(in: p0, p1) default(none) firstprivate(novo_n, prod_01, _prod_jmp) shared(p0, p1)
        {
            somar_mat(novo_n, p0, novo_n, p1, novo_n, prod_01, _prod_jmp);
        }        

        // C10 = P2 + P3
        #pragma omp task depend(in: p2, p3) default(none) firstprivate(novo_n, prod_10, _prod_jmp) shared(p2, p3)
        {
            somar_mat(novo_n, p2, novo_n, p3, novo_n, prod_10, _prod_jmp);
        }        

        // C11 = P4 + P0 - P2 - P6
        #pragma omp task depend(in: p4, p0, p2, p6) default(none) firstprivate(novo_n, prod_11, _prod_jmp) shared(p4, p0, p2, p6)
        {
            somar_mat(novo_n, p4, novo_n, p0, novo_n, prod_11, _prod_jmp);
            subtrair_mat(novo_n, prod_11, _prod_jmp, p2, novo_n, prod_11, _prod_jmp);
            subtrair_mat(novo_n, prod_11, _prod_jmp, p6, novo_n, prod_11, _prod_jmp);
        }        
    } // barreira implícita

    free(p0);
    free(p1);
    free(p2);
    free(p3);
    free(p4);
    free(p5);
    free(p6);
}

void mm_otimizado(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, 
                  double* _prod, long _prod_jmp) {
    double* _mat2_t = NULL;
    alocar_mem(&_mat2_t, _n*_n * sizeof(double));

    transpor_mat(_n, _mat2, _mat2_jmp, _mat2_t, _n);

    long num = _n / BLOCO_TAM;

    #pragma omp taskloop
    for (long i = 0; i < num; ++i) {
        for (long j = 0; j < num; ++j) {
            for (int k = 0; k < BLOCO_TAM; ++k) {
                double* p_prod = _prod + i*BLOCO_TAM*_prod_jmp + j*BLOCO_TAM + k*_prod_jmp;
                for (int m = 0; m < BLOCO_TAM; ++m) {
                    double soma = 0.0;
                    for (long r = 0; r < num; ++r) {
                        double* p_mat1 = _mat1 + i*BLOCO_TAM*_mat1_jmp + r*BLOCO_TAM + k*_mat1_jmp;
                        double* p_mat2_t = _mat2_t + j*BLOCO_TAM*_n + r*BLOCO_TAM + m*_n;
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

    free(_mat2_t);
}

void transpor_mat(long _n, double* _mat, long _mat_jmp, double* _mat_t, long _mat_t_jmp) {
    #pragma omp taskloop
    for (long i = 0; i < _n; ++i) {
        double* p_mat = _mat + i;
        double* p_mat_t = _mat_t + i*_mat_t_jmp;        
        for (long j = 0; j < _n; ++j) {
            *p_mat_t = *p_mat;            
            p_mat += _mat_jmp;
            p_mat_t++;
        }
    }
}

void somar_mat(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, 
               double* _adic, long _adic_jmp) {
    #pragma omp taskloop
    for (long i = 0; i < _n; ++i) {
        double* p_mat1 = _mat1 + i*_mat1_jmp;
        double* p_mat2 = _mat2 + i*_mat2_jmp;
        double* p_adic = _adic + i*_adic_jmp;
        for (long j = 0; j < _n; ++j) {
            *p_adic = *p_mat1 + *p_mat2;
            p_mat1++;
            p_mat2++;
            p_adic++;
        }        
    }
}

void subtrair_mat(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, 
                  double* _subt, long _subt_jmp) {
    #pragma omp taskloop
    for (long i = 0; i < _n; ++i) {
        double* p_mat1 = _mat1 + i*_mat1_jmp;
        double* p_mat2 = _mat2 + i*_mat2_jmp;
        double* p_subt = _subt + i*_subt_jmp;
        for (long j = 0; j < _n; ++j) {
            *p_subt = *p_mat1 - *p_mat2;
            p_mat1++;
            p_mat2++;
            p_subt++;
        }        
    }
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
