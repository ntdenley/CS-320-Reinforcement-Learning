#define DTYPE float

typedef struct {
    DTYPE * data;
    int nelem;
    int links;
} Storage;

Storage * initStorage(void);
void mallocData(Storage * s, int nelem);
void rmStorage(Storage * s);
void freeData(Storage * s);