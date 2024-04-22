import ml.array as ml
import ml.nn as nn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = ml.Array(X_train.tolist())
    X_test = ml.Array(X_test.tolist())
    y_train = ml.Array(y_train.tolist())
    y_test = ml.Array(y_test.tolist())
    return X_train, X_test, y_train, y_test

def train(lr, iter, reports=10):
    X_train, X_test, y_train, y_test = get_data()
    one_hot_train = nn.one_hot_encode(y_train, 3)
    
    W = ml.rand([4,3])
    b = ml.rand([3])

    logits = X_train @ W + b
    loss = nn.softmax_cross_entropy_loss(logits, one_hot_train) 
    loss.build_backward()
    optim = nn.Adam(lr, [W, b])

    for it in range(iter):
        if (it % (iter//reports) == 0): print(f"{it} loss = ", loss.eval().item())
        loss.eval()
        loss.zero_grad()
        loss.set_reeval()
        W.grad().set_reeval()
        b.grad().set_reeval()
        optim.step()
    
    logits_test = X_test.eval() @ W.eval() + (b.eval())
    predictions = np.argmax(logits_test, axis=1)
    accuracy = np.mean((predictions == y_test.eval()))
    print(f'Accuracy: {accuracy.item()}')
    loss.view_graph(view=False)

lr = 0.01
iter = 1000

train(lr,iter)