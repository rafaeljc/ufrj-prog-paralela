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
long* map = NULL;

void iniciar();
void iniciar_mapa();

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
        #pragma omp parallel for default(none) private(j) shared(A, Anew, map) reduction(max:dt)
        for (i = 1; i <= ROWS; i++) {
            for (j = 1; j <= COLUMNS; j++) {
                Anew[map[i*(COLUMNS+2)+j]] = 0.25 * (A[map[(i+1)*(COLUMNS+2)+j]] + A[map[(i-1)*(COLUMNS+2)+j]] + A[map[i*(COLUMNS+2)+j+1]] + A[map[i*(COLUMNS+2)+j-1]]);
                dt = fmax(fabs(Anew[map[i*(COLUMNS+2)+j]] - A[map[i*(COLUMNS+2)+j]]), dt);
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
    free(map);

    return 0;
}

void iniciar(){
    int i, j;

    iniciar_mapa();

    size_t size = (ROWS+2) * (COLUMNS+2) * sizeof(double);
    A = (double*) malloc(size);
    Anew = (double*) malloc(size);
    if (!A || !Anew) {
        fprintf(stderr, "Não foi possível alocar memória!\n");
        exit(1);
    }

    for (i = 1; i <= ROWS; i++) {
        for (j = 1; j <= COLUMNS; j++) {
            A[map[i*(COLUMNS+2)+j]] = 0.0;
        }
    }

    for (i = 1; i <= ROWS; i++) {
        A[map[i*(COLUMNS+2)]] = 0.0;
        Anew[map[i*(COLUMNS+2)]] = 0.0;
        A[map[i*(COLUMNS+2)+COLUMNS+1]] = (100.0/ROWS)*i;        
        Anew[map[i*(COLUMNS+2)+COLUMNS+1]] = (100.0/ROWS)*i;
    }

    for (j = 0; j <= COLUMNS+1; j++) {
        A[map[j]] = 0.0;
        Anew[map[j]] = 0.0;
        A[map[(ROWS+1)*(COLUMNS+2)+j]] = (100.0/COLUMNS)*j;        
        Anew[map[(ROWS+1)*(COLUMNS+2)+j]] = (100.0/COLUMNS)*j;
    }
}

void iniciar_mapa() {
    size_t size = (ROWS+2) * (COLUMNS+2) * sizeof(long);
    map = (long*) malloc(size);
    if (!map) {
        fprintf(stderr, "Não foi possível alocar memória!\n");
        exit(1);
    }

    long m = 0;
    for (int _i = 0; _i < ROWS+2; _i += 3) {
        int i_limit = _i + 3;
        if (i_limit > ROWS+2)
            i_limit = ROWS+2;
        for (int j = 0; j < COLUMNS+2; j++) {
            if (j % 2 == 0) {
                for (int i = _i; i < i_limit; i++) {
                    map[i*(COLUMNS+2)+j] = m;
                    m++;
                }
            }
            else {
                for (int i = i_limit-1; i >= _i; i--) {
                    map[i*(COLUMNS+2)+j] = m;
                    m++;
                }
            }
        }
    }
}
