#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#include "mpi.h"

// valores default
#define N 23  // quantidade de elementos do vetor (inteiro > 0)
#define K 100  // valor máximo de um número do vetor (inteiro > 0)

void imprime_vetor(int rank, int8_t* vetor, int size);
void imprime_msg_erro_memoria(int rank, char* var);
void prep_encerra_processo();
void trata_args_entrada(int argc, char* argv[], int* n, int* k);
void imprime_msg_arg_invalido(char* nome_arquivo);

// declaradas globalmente para facilitar as chamadas
// do free() na função prep_encerra_processo()
int8_t* vetor = NULL;
int8_t* vetor_parte = NULL;
int* scount = NULL;
int* displs = NULL;
int* aux_local = NULL;
int* aux_global = NULL;


int main(int argc, char* argv[]) {
    int meu_rank = -1;
    int num_proc = -1;
    int n = N;
    int k = K;
    double tempo_inicial = 0.0;
    double tempo_final = 0.0;

    trata_args_entrada(argc, argv, &n, &k);

    // rotinas de inicialização
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &meu_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    if (n < num_proc) {
        printf("O valor de 'n' precisa ser maior ou igual ao número de processos!\n");
        prep_encerra_processo();
        exit(3);
    }

    tempo_inicial = MPI_Wtime();

    // calcula distribuição de carga
    scount = (int*) malloc(num_proc * sizeof(int));
    if (!scount) {
        imprime_msg_erro_memoria(meu_rank, "scount");
        prep_encerra_processo();
        exit(1);
    }
    if (meu_rank == 0) {
        int div_trab = n / num_proc;
        for (int i = 0; i < num_proc; ++i)
            scount[i] = div_trab;
        int res = n % num_proc;
        for (int i = 0; i < res; ++i)
            ++scount[i];
    }
    MPI_Request pedido_bcast;
    MPI_Ibcast(scount, num_proc, MPI_INT, 0, MPI_COMM_WORLD, &pedido_bcast);

    if (meu_rank == 0) {
        // inicia vetor com números inteiros [0, k)
        vetor = (int8_t*) malloc(n * sizeof(int8_t));
        if (!vetor) {
            imprime_msg_erro_memoria(meu_rank, "vetor");
            prep_encerra_processo();
            exit(1);
        }       
        srand(time(NULL));
        for (int i = 0; i < n; ++i)
            vetor[i] = rand() % (k + 1);
        imprime_vetor(meu_rank, vetor, n);
    }
    MPI_Wait(&pedido_bcast, MPI_STATUS_IGNORE);
    vetor_parte = (int8_t*) malloc(scount[meu_rank] * sizeof(int8_t));
    if (!vetor_parte) {
        imprime_msg_erro_memoria(meu_rank, "vetor_parte");
        prep_encerra_processo();
        exit(1);
    }

    // divide o vetor entre os processos
    if (meu_rank == 0) {
        displs = (int*) malloc(num_proc * sizeof(int));
        if (!displs) {
            imprime_msg_erro_memoria(meu_rank, "displs");
            prep_encerra_processo();
            exit(1);
        }
        displs[0] = 0;
        for (int i = 1; i < num_proc; ++i)
            displs[i] = displs[i - 1] + scount[i - 1];
    }
    MPI_Request pedido_scatterv;
    MPI_Iscatterv(vetor, scount, displs, MPI_INT8_T, vetor_parte, scount[meu_rank], MPI_INT8_T, 0, MPI_COMM_WORLD, &pedido_scatterv);
    
    // vetor auxiliar para execução do counting sort
    aux_local = (int*) calloc((k + 1), sizeof(int));
    if (!aux_local) {
        imprime_msg_erro_memoria(meu_rank, "aux_local");
        prep_encerra_processo();
        exit(1);
    }

    // conta os valores do vetor
    MPI_Wait(&pedido_scatterv, MPI_STATUS_IGNORE);
    for (int i = 0; i < scount[meu_rank]; ++i)
        ++aux_local[vetor_parte[i]];

    // consolida o vetor aux entre todos os processos
    aux_global = (int*) malloc((k + 1) * sizeof(int));
    if (!aux_global) {
        imprime_msg_erro_memoria(meu_rank, "aux_global");
        prep_encerra_processo();
        exit(1);
    }
    MPI_Request pedido_allreduce;
    MPI_Iallreduce(aux_local, aux_global, (k + 1), MPI_INT, MPI_SUM, MPI_COMM_WORLD, &pedido_allreduce);

    // prepara para fase final do counting sort
    // 1) calcula ajuste a esquerda
	int qnt_esq = 0;
	for (int p = 0; p < num_proc; ++p) {
		if (p < meu_rank)
			qnt_esq += scount[p];
	}
	// 2) ajuste a esquerda
    MPI_Wait(&pedido_allreduce, MPI_STATUS_IGNORE);
	int i = 0;
	while (qnt_esq > 0) {
		if (qnt_esq >= aux_global[i]) {
			qnt_esq -= aux_global[i];
			aux_global[i] = 0;
			++i;
		} else {
			aux_global[i] -= qnt_esq;
			qnt_esq = 0;
		}
	}
	// 3) ajuste no meio
    int qnt_meio = scount[meu_rank];
	while (qnt_meio > 0) {
		if (qnt_meio >= aux_global[i]) {
			qnt_meio -= aux_global[i];
		} else {
			aux_global[i] = qnt_meio;
			qnt_meio = 0;
		}
		++i;
	}
	// 4) ajuste a direita
	while (i < num_proc) {
		aux_global[i] = 0;
		++i;
	}

    // fase final do counting sort
    i = 0;
	for (int j = 0; j <= k; ++j) {
		while (aux_global[j] > 0) {
			vetor_parte[i] = j;
			--aux_global[j];
			++i;
		}
	}

    // monta o vetor ordenado no processo raiz
    MPI_Request pedido_gatherv;
    MPI_Igatherv(vetor_parte, scount[meu_rank], MPI_INT8_T, vetor, scount, displs, MPI_INT8_T, 0, MPI_COMM_WORLD, &pedido_gatherv);
    MPI_Wait(&pedido_gatherv, MPI_STATUS_IGNORE);

    tempo_final = MPI_Wtime();

    if (meu_rank == 0) {
        imprime_vetor(meu_rank, vetor, n);
        printf("Número de processos: %d\n", num_proc);
        printf("n = %d\n", n);
        printf("k = %d\n", k);
        printf("Tempo de execução: %1.10f\n", tempo_final - tempo_inicial);
    } 

    prep_encerra_processo();
    MPI_Finalize();
    return 0;
}


void imprime_vetor(int rank, int8_t* vet, int size) {
    printf("[%d]: [ ", rank);
    for (int i = 0; i < size; ++i)
        printf("%d, ", vet[i]);
    printf("]\n");
}

void imprime_msg_erro_memoria(int rank, char* var) {
    fprintf(stderr, "[%d]: Não foi possível alocar memória (%s)\n", rank, var);
}

void prep_encerra_processo() {
    if (vetor != NULL)
        free(vetor);
    if (vetor_parte != NULL)
        free(vetor_parte);
    if (scount != NULL)
        free(scount);
    if (displs != NULL)
        free(displs);
    if (aux_local != NULL)
        free(aux_local);
    if (aux_global != NULL)
        free(aux_global);
}

void trata_args_entrada(int argc, char* argv[], int* n, int* k) {
    // argumentos opcionais
    if (argc > 1) {    
        for (int i = 1; i < argc; i += 2) {
            // verifica se existe o valor do parâmetro
            if (i + 1 >= argc)
                imprime_msg_arg_invalido(argv[0]);
            
            if (strcmp(argv[i], "--n") == 0) {
                *n = atoi(argv[i + 1]);
                if (*n < 1)
                    imprime_msg_arg_invalido(argv[0]);
                continue;
            }
            if (strcmp(argv[i], "--k") == 0) {
                *k = atoi(argv[i + 1]);
                if (*k < 1)
                    imprime_msg_arg_invalido(argv[0]);
                continue;
            }        
            // caso não seja nenhum dos parâmetros acima
            imprime_msg_arg_invalido(argv[0]);
        }
    }
}

void imprime_msg_arg_invalido(char* nome_arquivo) {
    printf("Argumento(s) inválido(s)!\n");
    printf("Uso: %s [--n <num>] [--k <num>]\n", nome_arquivo);
    printf("\tn > 0 (int)\n");
    printf("\tk > 0 (int)\n");
    exit(2);
}
