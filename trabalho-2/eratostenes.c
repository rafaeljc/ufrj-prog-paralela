#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

int eratostenes(int n);


int main(int argc, char* argv[]) {
    int n = -1;
    int qnt_primos = -1;

    if (argc == 2) {
        n = atoi(argv[1]);
    } else {
        fprintf(stderr, "Número de argumentos inválido!\n");
        exit(1);
    }    

    double t_inicio = omp_get_wtime();

    qnt_primos = eratostenes(n);

    double t_fim = omp_get_wtime();

    printf("Existem %d números primos entre 2 e %d\n", qnt_primos, n);
    printf("Tempo de execução: %.15lf\n", t_fim - t_inicio);

    return 0;
}


int eratostenes(int n) {
    if (n < 2)
        return 0;

    size_t size = (n + 1) * sizeof(bool);
    bool* eh_primo = (bool*) malloc(size);
    if (eh_primo == NULL) {
        fprintf(stderr, "Não foi possível alocar memória!\n");
        exit(2);
    }

    #pragma omp parallel for default(none) shared(n, eh_primo) schedule(static)
    for (int i = 0; i < n + 1; ++i) {
        eh_primo[i] = true;
    } // barreira implícita

    for (int i = 2; i*i <= n; ++i) {
        if (eh_primo[i]) {
            #pragma omp parallel for default(none) shared(n, i, eh_primo) schedule(static)
            for (int j = i*i; j <= n; j += i) {
                eh_primo[j] = false;
            } // barreira implícita            
        }            
    }        

    int qnt_primos = 0;

    #pragma omp parallel for default(none) shared(n, eh_primo) reduction(+:qnt_primos) schedule(static)
    for (int i = 3; i <= n; i += 2) {
        if (eh_primo[i])
            ++qnt_primos;
    } // barreira implícita
    qnt_primos++; // contando o 2

    free(eh_primo);

    return qnt_primos;
}
