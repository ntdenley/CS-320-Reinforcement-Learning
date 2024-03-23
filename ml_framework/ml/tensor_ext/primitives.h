/*
- sum_simple
    Storage
- max_simple
    Storage
- sum_over_dim
    Storage shape stride
- max_over_dim
    Storage shape stride

shape
    contiguous
- reshape
    non-contiguous
- transpose
- broadcast
*/

#define FastLoop(Start, End, Expr) {\
    int i; \
    for (i = Start; i < End; i++) { \
        Expr \
    } \
}

#define Map(IndexedStorage, Expr) { \
    int _end = IndexedStorage->nelem; \
    DTYPE * _data = IndexedStorage->data; \
    FastLoop(0, _end, _data[i] = Expr;);\
}

#define BinaryOpContiguous(IndexedStorage, Expr) { \
    printf("BinaryOpContiguous not defined yet!\n");\
    exit();\
}

#define BinaryOpNonContiguous(IndexedStorage, Expr) { \
    printf("BinaryOpNonContiguous not defined yet!\n");\
    exit();\
}

#define Matmul() { \
    printf("Matmul not defined yet!\n");\
    exit();\
}