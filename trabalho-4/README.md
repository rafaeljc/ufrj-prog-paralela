## mm_ingenuo
```console
gcc mm_ingenuo.c -O1 -fopenmp -o mm_ingenuo
```

## mm_otimizado
```console
gcc mm_otimizado.c -O3 -fno-tree-loop-vectorize -fno-tree-slp-vectorize -fno-tree-vectorize -fopenmp -o mm_otimizado
```

## mm_strassen_paralelo
```console
gcc mm_strassen_paralelo.c -O3 -fno-tree-loop-vectorize -fno-tree-slp-vectorize -fno-tree-vectorize -fopenmp -o mm_strassen_paralelo
```

## mm_strassen_paralelo (com vetorização)
```console
gcc mm_strassen_paralelo.c -O3 -mavx512f -mavx512er -mavx512cd -mavx512pf -fopenmp -o mm_strassen_paralelo_vet
```