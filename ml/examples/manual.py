import ml.array as ml
import ml.nn as nn

a = ml.Array([1,2,3])
b = ml.Array([100,200,300])

out = (a + b).sqrt() 

out.eval()