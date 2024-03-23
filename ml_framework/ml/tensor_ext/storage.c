#include <stdlib.h>
#include "storage.h"

Storage * initStorage(void) {
    Storage * s = malloc(sizeof(Storage));
    s->data = NULL;
    s->nelem = -1;
    s->links = 0;
    return s;
}

void mallocData(Storage * s, int nelem) {
    s->nelem = nelem;
    s->data = malloc(sizeof(DTYPE) * nelem);
}

void freeData(Storage * s) {
    s->nelem = -1;
    free(s->data);
}

void rmStorage(Storage * s) {
    free(s);
}