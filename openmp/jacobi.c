/*
    código original: https://github.com/gpsilva2003/OPENMP/blob/main/src/omp_jacobi.c
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define COLUMNS 2000
#define ROWS 2000
#define MAX_TEMP_ERROR 0.01

double* A = NULL;
double* Anew = NULL;

void iniciar();

int main(int argc, char* argv[]) {
    int i, j;
    int max_iterations = 3000;
    int iteration = 1;
    double dt = 100;

    iniciar();

    double inicio = omp_get_wtime();

    double* tmp = NULL;
    while (dt > MAX_TEMP_ERROR && iteration <= max_iterations) {
        dt = 0.0;
        #pragma omp parallel for default(none) private(j) shared(A, Anew) reduction(max:dt)
        for (i = 1; i <= ROWS; i++) {
            for (j = 1; j <= COLUMNS; j++) {
                Anew[i*(COLUMNS+2)+j] = 0.25 * (A[(i+1)*(COLUMNS+2)+j] + A[(i-1)*(COLUMNS+2)+j] + A[i*(COLUMNS+2)+j+1] + A[i*(COLUMNS+2)+j-1]);
                dt = fmax(fabs(Anew[i*(COLUMNS+2)+j] - A[i*(COLUMNS+2)+j]), dt);
            }
        } // barreira implícita        

        tmp = A;    
        A = Anew;
        Anew = tmp;

        iteration++;
    }

    double fim = omp_get_wtime();

    printf("Erro máximo na iteração %d era %f. O tempo de execução foi de %f segundos\n", (iteration - 1), dt, (fim - inicio));

    free(A);
    free(Anew);

    return 0;
}

void iniciar(){
    int i, j;

    size_t size = (ROWS+2) * (COLUMNS+2) * sizeof(double);
    A = (double*) malloc(size);
    Anew = (double*) malloc(size);
    if (!A || !Anew) {
        fprintf(stderr, "Não foi possível alocar memória!\n");
        exit(1);
    }

    for (i = 1; i <= ROWS; i++) {
        for (j = 1; j <= COLUMNS; j++) {
            A[i*(COLUMNS+2)+j] = 0.0;
        }
    }

    for (i = 1; i <= ROWS; i++) {
        A[i*(COLUMNS+2)] = 0.0;
        Anew[i*(COLUMNS+2)] = 0.0;
        A[i*(COLUMNS+2)+COLUMNS+1] = (100.0/ROWS)*i;        
        Anew[i*(COLUMNS+2)+COLUMNS+1] = (100.0/ROWS)*i;
    }

    for (j = 0; j <= COLUMNS+1; j++) {
        A[j] = 0.0;
        Anew[j] = 0.0;
        A[(ROWS+1)*(COLUMNS+2)+j] = (100.0/COLUMNS)*j;        
        Anew[(ROWS+1)*(COLUMNS+2)+j] = (100.0/COLUMNS)*j;
    }
}
