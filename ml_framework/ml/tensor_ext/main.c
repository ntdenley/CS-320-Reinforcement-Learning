#include "storage.h"

int main(void) {
    Storage * s = initStorage();
    rmStorage(s);
}