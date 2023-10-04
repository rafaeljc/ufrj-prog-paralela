#include <stdio.h>
#define N 1000000

#include "mpi.h"


int main(int argc, char *argv[]) { /* calcpi_seq.c  */
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

    double pi = 0.0f;
    for (long i = meu_rank; i < N; i += num_proc) {
         double t = (double) ((i+0.5)/N);
         pi += 4.0/(1.0+t*t);
    }

    if (meu_rank == 0)
        MPI_Ireduce(MPI_IN_PLACE, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, &pedido);
    else
        MPI_Ireduce(&pi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, &pedido);
    MPI_Wait(&pedido, &estado);

    tempo_final = MPI_Wtime();

    if (meu_rank == 0) {
        printf("n = %d\n", N);
        printf("pi = %f\n", pi/N);
        printf("Tempo de execução: %1.10f\n", tempo_final - tempo_inicial);
    }        
    
    MPI_Finalize();
    return 0;
}
