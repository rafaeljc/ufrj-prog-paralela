#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "mpi.h"

void imprime_vetor(int rank, int* vetor, int size);
void imprime_msg_erro_memoria(int rank, char* nome_var);
void prep_encerra_processo();
void trata_args_entrada(int argc, char* argv[], int* n, int* k);
void imprime_msg_arg_invalido(char* nome_arquivo);

// declaradas globalmente para facilitar as chamadas
// do free() na função prep_encerra_processo()
int* scounts = NULL;
int* displs = NULL;
int* vetor = NULL;
int* aux = NULL;

int main(int argc, char* argv[]) {
    int meu_rank = -1;
    int num_proc = -1;
    MPI_Request pedido;
    MPI_Status estado;
    double tempo_inicial = 0.0;
    double tempo_final = 0.0;

    // valores default
    int n = 23;
    int k = 100;

    // rotinas de inicialização
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &meu_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    tempo_inicial = MPI_Wtime();

    if (meu_rank == 0)
        trata_args_entrada(argc, argv, &n, &k);
    MPI_Ibcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD, &pedido);
    MPI_Wait(&pedido, &estado);
    if (n < num_proc) {
        printf("O valor de 'n' precisa ser maior ou igual ao número de processos!\n");
        prep_encerra_processo();
        exit(3);        
    }
    MPI_Ibcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD, &pedido);
    MPI_Wait(&pedido, &estado);
    //printf("[%d]: n = %d / k = %d\n", meu_rank, n, k);

    // aloca memória para os vetores do counting sort
    vetor = (int*) malloc(n * sizeof(int));
    if (!vetor) {
        imprime_msg_erro_memoria(meu_rank, "vetor");
        prep_encerra_processo();
        exit(1);
    } 
    aux = (int*) calloc((k + 1), sizeof(int));
    if (!aux) {
        imprime_msg_erro_memoria(meu_rank, "aux");
        prep_encerra_processo();
        exit(1);
    }

    // preenche o vetor com valores inteiros aleatórios [0, k]
    if (meu_rank == 0) {
        srand(time(NULL));
        for (int i = 0; i < n; ++i)
            vetor[i] = rand() % (k + 1);
        //imprime_vetor(meu_rank, vetor, n);
    }
    //imprime_vetor(meu_rank, vetor, n);

    // distribuição de carga
    scounts = (int*) malloc(num_proc * sizeof(int));
    if (!scounts) {
        imprime_msg_erro_memoria(meu_rank, "scounts");
        prep_encerra_processo();
        exit(1);
    }
    displs = (int*) malloc(num_proc * sizeof(int));
    if (!displs) {
        imprime_msg_erro_memoria(meu_rank, "displs");
        prep_encerra_processo();
        exit(1);
    }
    if (meu_rank == 0) {
        int div_trab = n / num_proc;
        for (int i = 0; i < num_proc; ++i)
            scounts[i] = div_trab;
        int res = n % num_proc;
        for (int i = 0; i < res; ++i)
            ++scounts[i];
        displs[0] = 0;
        for (int i = 1; i < num_proc; ++i)
            displs[i] = displs[i - 1] + scounts[i - 1];
    }
    MPI_Ibcast(scounts, num_proc, MPI_INT, 0, MPI_COMM_WORLD, &pedido);
    MPI_Wait(&pedido, &estado);
    //imprime_vetor(meu_rank, scounts, num_proc);
    MPI_Ibcast(displs, num_proc, MPI_INT, 0, MPI_COMM_WORLD, &pedido);
    MPI_Wait(&pedido, &estado);
    //imprime_vetor(meu_rank, displs, num_proc);

    if (meu_rank == 0)
        MPI_Iscatterv(vetor, scounts, displs, MPI_INT, MPI_IN_PLACE, scounts[meu_rank], MPI_INT, 0, MPI_COMM_WORLD, &pedido);
    else
        MPI_Iscatterv(vetor, scounts, displs, MPI_INT, &vetor[displs[meu_rank]], scounts[meu_rank], MPI_INT, 0, MPI_COMM_WORLD, &pedido);
    MPI_Wait(&pedido, &estado);
    //imprime_vetor(meu_rank, vetor, n);

    // conta os valores do vetor
    int* p_vetor = &vetor[displs[meu_rank]];
    for (int i = 0; i < scounts[meu_rank]; ++i)
        ++aux[p_vetor[i]];
    //imprime_vetor(meu_rank, aux, (k + 1));

    // consolida o vetor aux entre todos os processos
    MPI_Iallreduce(MPI_IN_PLACE, aux, (k + 1), MPI_INT, MPI_SUM, MPI_COMM_WORLD, &pedido);
    MPI_Wait(&pedido, &estado);
    //imprime_vetor(meu_rank, aux, (k + 1));

    // prepara para fase final do counting sort
    // 1) calcula ajuste a esquerda
	int qnt_esq = 0;
	for (int i = 0; i < num_proc; ++i) {
		if (i < meu_rank)
			qnt_esq += scounts[i];
	}
	// 2) ajuste a esquerda
	int i = 0;
	while (qnt_esq > 0) {
		if (qnt_esq >= aux[i]) {
			qnt_esq -= aux[i];
			aux[i] = 0;
			++i;
		} else {
			aux[i] -= qnt_esq;
			qnt_esq = 0;
		}
	}
	// 3) ajuste no meio
    int qnt_meio = scounts[meu_rank];
	while (qnt_meio > 0) {
		if (qnt_meio >= aux[i]) {
			qnt_meio -= aux[i];
		} else {
			aux[i] = qnt_meio;
			qnt_meio = 0;
		}
		++i;
	}
	// 4) ajuste a direita
	while (i <= k) {
		aux[i] = 0;
		++i;
	}
    //imprime_vetor(meu_rank, aux, (k + 1));

    // fase final do counting sort
    i = 0;
    p_vetor = &vetor[displs[meu_rank]];
	for (int j = 0; j <= k; ++j) {
		while (aux[j] > 0) {
			p_vetor[i] = j;
			--aux[j];
			++i;
		}
	}
    //imprime_vetor(meu_rank, vetor, n);

    // monta o vetor ordenado no processo raiz
    if (meu_rank == 0)
        MPI_Igatherv(MPI_IN_PLACE, scounts[meu_rank], MPI_INT, vetor, scounts, displs, MPI_INT, 0, MPI_COMM_WORLD, &pedido);
    else
        MPI_Igatherv(&vetor[displs[meu_rank]], scounts[meu_rank], MPI_INT, vetor, scounts, displs, MPI_INT, 0, MPI_COMM_WORLD, &pedido);
    MPI_Wait(&pedido, &estado);

    tempo_final = MPI_Wtime();

    if (meu_rank == 0) {
        //imprime_vetor(meu_rank, vetor, n);
        printf("Número de processos: %d\n", num_proc);
        printf("n = %d\n", n);
        printf("k = %d\n", k);
        printf("Tempo de execução: %1.10f\n", tempo_final - tempo_inicial);
    } 

    prep_encerra_processo();
    MPI_Finalize();
    return 0;
}


void imprime_vetor(int rank, int* vet, int size) {
    printf("[%d]: [ ", rank);
    for (int i = 0; i < size; ++i)
        printf("%d, ", vet[i]);
    printf("]\n");
}

void imprime_msg_erro_memoria(int rank, char* nome_var) {
    printf("[%d]: Não foi possível alocar memória (%s)\n", rank, nome_var);
}

void prep_encerra_processo() {
    if (scounts != NULL)
        free(scounts);
    if (displs != NULL)
        free(displs);
    if (vetor != NULL)
        free(vetor);
    if (aux != NULL)
        free(aux);
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
