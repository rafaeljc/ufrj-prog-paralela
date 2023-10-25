#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

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

    qnt_primos = eratostenes(n);

    printf("Existem %d números primos entre 2 e %d\n", qnt_primos, n);
    
    return 0;
}


int eratostenes(int n) {
    if (n < 2)
        return 0;
        
    int qnt_primos = 1; // 2

    size_t size = (n + 1) * sizeof(bool);
    bool* eh_primo = (bool*) malloc(size);
    if (eh_primo == NULL) {
        fprintf(stderr, "Não foi possível alocar memória!\n");
        exit(2);
    }

    memset(eh_primo, true, size);

    for (int i = 2; i*i <= n; ++i)
        if (eh_primo[i])
            for (int j = i*i; j <= n; j += i)
                eh_primo[j] = false;

    for (int i = 3; i <= n; i+= 2)
        if (eh_primo[i])
            ++qnt_primos;

    free(eh_primo);

    return qnt_primos;
}
