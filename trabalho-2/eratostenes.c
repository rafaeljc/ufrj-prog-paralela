#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

long eratostenes(long n);


int main(int argc, char* argv[]) {
    long n = -1;
    long qnt_primos = -1;

    if (argc == 2) {
        n = atol(argv[1]);
    } else {
        fprintf(stderr, "Número de argumentos inválido!\n");
        exit(1);
    }    

    double t_inicio = omp_get_wtime();

    qnt_primos = eratostenes(n);

    double t_fim = omp_get_wtime();

    printf("Existem %ld números primos entre 2 e %ld\n", qnt_primos, n);
    printf("Tempo de execução: %.15lf\n", t_fim - t_inicio);

    return 0;
}


long eratostenes(long n) {
    if (n < 2)
        return 0;

    size_t size = (n + 1) * sizeof(bool);
    bool* eh_primo = (bool*) malloc(size);
    if (eh_primo == NULL) {
        fprintf(stderr, "Não foi possível alocar memória!\n");
        exit(2);
    }

    #pragma omp parallel for default(none) shared(n, eh_primo) schedule(static)
    for (long i = 0; i < n + 1; ++i) {
        eh_primo[i] = true;
    } // barreira implícita

    for (long i = 2; i*i <= n; ++i) {
        if (eh_primo[i]) {
            #pragma omp parallel for default(none) shared(n, i, eh_primo) schedule(static)
            for (long j = i*i; j <= n; j += i) {
                eh_primo[j] = false;
            } // barreira implícita            
        }            
    }        

    long qnt_primos = 0;

    #pragma omp parallel for default(none) shared(n, eh_primo) reduction(+:qnt_primos) schedule(static)
    for (long i = 3; i <= n; i += 2) {
        if (eh_primo[i])
            ++qnt_primos;
    } // barreira implícita
    qnt_primos++; // contando o 2

    free(eh_primo);

    return qnt_primos;
}
