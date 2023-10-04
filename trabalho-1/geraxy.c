#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define RANGE_BEGIN -100.0
#define RANGE_END 100.0
#define DESVIO_MIN -2.5
#define DESVIO_MAX 2.5

double f(double x) {
    return 2.0*x + 3.0;
}


int main(int argc, char* argv[]) {
    int n = 0;
    if (argc != 2) {
        printf("Falta argumento ou argumento inválido!\n");
        printf("Uso: %s <num>\n", argv[0]);
        printf("\t<num> inteiro > 0\n");
        return 1;
    } else {
        n = atoi(argv[1]);
    }

    FILE* arquivo;
    arquivo = fopen("xydata", "w");
    if (arquivo == NULL) {
        printf("Não foi possível abrir o arquivo!\n");
        return 2;
    }

    fprintf(arquivo, "%d ", n);

    srand(time(NULL));
    double h = (RANGE_END - RANGE_BEGIN) / n;
    for (double x = RANGE_BEGIN; x <= RANGE_END; x += h) {
        double y = f(x);
        double desvio = rand() / (double) RAND_MAX;  // [0.0, 1.0]
        desvio = DESVIO_MIN + desvio*(DESVIO_MAX - DESVIO_MIN);  // [DESVIO_MIN, DESVIO_MAX]
        y *= 1.0 + desvio/100.0;
        fprintf(arquivo, "%lf %lf ", x, y);
    }

    fclose(arquivo);

    return 0;
}
