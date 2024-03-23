#include <vector>
using namespace std;

#include "storage.h"

struct TensorBase {
   Storage * storage;
   int ndim;
   vector<int> shape;
   vector<int> stride; 
};