import torch
import ml.array as ml
import ml.nn as nn
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def one_hot_encode(data, classes):
    out = []
    for d in data:
        one_hot = [0 for _ in range(classes)]
        one_hot[int(d)] = 1
        out.append(one_hot)
    return out

def softmax_cross_entropy_loss(logits, targets):
    s = logits.shape[0] if isinstance(logits, torch.Tensor) else logits.node.shape[0]
    loss = -(targets * logits.softmax(1).log()).sum() / s
    return loss

def get_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_data() 

w_start = torch.rand((4, 3)).tolist()
b_start = torch.rand(3).tolist()

def torch_version(lr, iter, reports=10):
    # X_train, X_test, y_train, y_test = get_data()
    one_hot_train = torch.tensor(one_hot_encode(y_train,3))
    
    # Initialize weights and biases
    W = torch.tensor(w_start, requires_grad=True)
    b = torch.tensor(b_start, requires_grad=True)
    logits = X_train @ W + b

    for epoch in range(iter):
        logits = X_train @ W + b
        loss = softmax_cross_entropy_loss(logits, one_hot_train)
        loss.backward()
        with torch.no_grad():
            W -= lr * W.grad
            b -= lr * b.grad
        W.grad.zero_()
        b.grad.zero_()
        if epoch % (iter//reports) == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    with torch.no_grad():
        logits_test = X_test.mm(W) + b
        # print(logits_test[:10])
        predictions = torch.argmax(logits_test, dim=1)
        accuracy = torch.mean((predictions == y_test).float())
        print(f'Accuracy: {accuracy.item()}')
    
def my_version(lr, iter, reports=10):
    out = [ml.Array(x.tolist()) for x in [X_train, X_test, y_train, y_test]] 
    X_train1, X_test1, y_train1, y_test1 = out[0], out[1], out[2], out[3]
    one_hot_train = ml.Array(one_hot_encode(y_train1.eval().tolist(),3))
    
    # Initialize weights and biases
    W = ml.Array(w_start)
    b = ml.Array(b_start)

    logits = X_train1 @ W + b
    loss = softmax_cross_entropy_loss(logits, one_hot_train) 
    loss.build_backward()
    optim = nn.SGD(lr, [W, b])

    for it in range(iter):
        if (it % (iter//reports) == 0): print(f"{it} loss = ", loss.eval().item())
        loss.eval()
        loss.zero_grad()
        loss.set_reeval()
        W.grad().set_reeval()
        b.grad().set_reeval()
        optim.step()
    
    W = torch.tensor(W.eval())
    b = torch.tensor(b.eval())
    logits_test = X_test.mm(W) + b
    # print(logits_test[:10])
    predictions = torch.argmax(logits_test, dim=1)
    accuracy = torch.mean((predictions == y_test).float())
    print(f'Accuracy: {accuracy.item()}')
    loss.view_graph()

lr = 0.01
iter = 1000

print("     PYTORCH")
torch_version(lr,iter)

print("\n     ML FRAMEWORK")
my_version(lr,iter)