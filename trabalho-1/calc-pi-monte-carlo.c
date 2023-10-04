/**
 *
 * Autor: Adrian Statescu mergesortv@gmail.com http://adrianstatescu.com
 *
 * Descrição:  Programa em C Program para computar o valor de PI usando o método de Monte Carlo.
 *
 * Licença MIT 
 *
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "mpi.h"

#define SEED time(NULL)


int main(int argc, char *argv[]) {
    int i, count, n=1000000;
    double x,y,z,pi; 
    int meu_rank = -1;
    int num_proc = -1;
    MPI_Request pedido;
    MPI_Status estado;
    double tempo_inicial = 0.0;
    double tempo_final = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &meu_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    tempo_inicial = MPI_Wtime();

    srand( SEED );       
    count = 0;
    for (i = meu_rank; i < n; i += num_proc) {
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;
        z = x * x + y * y;
        if( z <= 1 ) count++;
    }

    if (meu_rank == 0)
        MPI_Ireduce(MPI_IN_PLACE, &count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD, &pedido);
    else
        MPI_Ireduce(&count, &count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD, &pedido);
    MPI_Wait(&pedido, &estado);

    pi = (double) count / n * 4;

    tempo_final = MPI_Wtime();

    if (meu_rank == 0) {
        printf("n = %d\n", n);
        printf("Aproximação de PI é = %g\n", pi);
        printf("Tempo de execução: %1.10f\n", tempo_final - tempo_inicial);
    } 

    MPI_Finalize();
    return 0;
}
