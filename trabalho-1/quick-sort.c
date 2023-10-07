#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>

#include "mpi.h"

#define N 11

int compara_int(const void* a, const void* b) {
   return *(int*)a - *(int*)b;
}

void imprime_vetor(int rank, int* vet, int size) {
    printf("[%d]: [ ", rank);
    for (int i = 0; i < size; ++i)
        printf("%d, ", vet[i]);
    printf("]\n");
}


int main(int argc, char* argv[]) {
    int meu_rank = -1;
    int num_proc = -1;
    int* vetor = NULL;
    int* resultado = NULL;
    int* scounts = NULL;
    int* displs = NULL;
    MPI_Request pedido;

    // rotinas de inicialização
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &meu_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    double tempo_inicial = MPI_Wtime();

    // distribuição de carga
    scounts = (int*) malloc(num_proc * sizeof(int));
    displs = (int*) malloc(num_proc * sizeof(int));
    int div_trab = N / num_proc;
    for (int i = 0; i < num_proc; ++i)
        scounts[i] = div_trab;
    int res = N % num_proc;
    for (int i = 0; i < res; ++i)
        ++scounts[i];
    displs[0] = 0;
    for (int i = 1; i < num_proc; ++i)
        displs[i] = displs[i - 1] + scounts[i - 1];

    if (meu_rank == 0) {
        vetor = (int*) malloc(N * sizeof(int));
        // preenche o vetor com valores inteiros aleatórios
        srand(time(NULL));
        for (int i = 0; i < N; ++i)
            vetor[i] = rand() % 101;
        //imprime_vetor(meu_rank, vetor, N);
        MPI_Iscatterv(vetor, scounts, displs, MPI_INT, MPI_IN_PLACE, scounts[meu_rank], MPI_INT, 0, MPI_COMM_WORLD, &pedido);
    } else {
        vetor = (int*) malloc(scounts[meu_rank] * sizeof(int));
        MPI_Iscatterv(vetor, scounts, displs, MPI_INT, vetor, scounts[meu_rank], MPI_INT, 0, MPI_COMM_WORLD, &pedido);
    }
    MPI_Wait(&pedido, MPI_STATUS_IGNORE);
    //imprime_vetor(meu_rank, vetor, scounts[meu_rank]);

    // ordena sua parte
    qsort(vetor, scounts[meu_rank], sizeof(int), compara_int);
    //imprime_vetor(meu_rank, vetor, scounts[meu_rank]);

    // envia vetores ordenados ao processo raiz
    if (meu_rank == 0)
        MPI_Igatherv(MPI_IN_PLACE, scounts[meu_rank], MPI_INT, vetor, scounts, displs, MPI_INT, 0, MPI_COMM_WORLD, &pedido);
    else
        MPI_Igatherv(vetor, scounts[meu_rank], MPI_INT, vetor, scounts, displs, MPI_INT, 0, MPI_COMM_WORLD, &pedido);
    MPI_Wait(&pedido, MPI_STATUS_IGNORE);

    // merge
    if (meu_rank == 0) {
        resultado = (int*) malloc(N * sizeof(int));
        for (int i = 0; i < N; ++i) {
            int menor = INT_MAX;
            int p_menor = -1;
            for (int p = 0; p < num_proc; ++p) {
                int end = p*scounts[0] + scounts[p];
                if (displs[p] >= end)
                    continue;
                int valor = vetor[displs[p]];
                if (valor < menor) {
                    menor = valor;
                    p_menor = p;
                }
            }
            resultado[i] = menor;
            ++displs[p_menor];
        }
    }    
    
    double tempo_final = MPI_Wtime();
    if (meu_rank == 0) {
        //imprime_vetor(meu_rank, resultado, N);
        printf("Número de processos: %d\n", num_proc);
        printf("n = %d\n", N);
        printf("Tempo de execução: %1.10f\n", tempo_final - tempo_inicial);
    }

    if (vetor != NULL)
        free(vetor);
    if (resultado != NULL)
        free(resultado);
    if (scounts != NULL)
        free(scounts);
    if (displs != NULL)
        free(displs);

    MPI_Finalize();
    return 0;
}
