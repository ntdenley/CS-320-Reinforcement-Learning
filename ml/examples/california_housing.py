import ml.array as ml
import ml.nn as nn
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_data():
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = ml.Array(X_train.tolist())
    X_test = ml.Array(X_test.tolist())
    y_train = ml.Array(y_train.tolist())
    y_test = ml.Array(y_test.tolist())
    return X_train, X_test, y_train, y_test

def mean_squared_error(predictions, targets):
    return ((predictions - targets).square()).sum() / ml.nelem(predictions.node.shape)

def batch_generator(X, y, batch_size):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)  # Shuffle indices to ensure random batches

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_idx = indices[start_idx:end_idx]
        yield X[batch_idx], y[batch_idx]

def train(lr, iter, reports=10, batch_size=64*2):
    X_train, X_test, y_train, y_test = get_data()
    
    W = ml.rand([X_train.node.shape[1], 1])  # Adjust dimensions for regression
    b = ml.rand([1])  # Single bias for one output

    for X_batch, y_batch in batch_generator(X_train.eval(), y_train.eval(), batch_size):
        values = ml.Array(X_batch.tolist())
        labels = ml.Array(y_batch.tolist())
        break
    logits = values @ W + b
    loss = mean_squared_error(logits, labels)  # Use MSE for regression
    loss.build_backward()
    optim = nn.Adam(lr, [W, b])

    for it in range(iter):
        for X_batch, y_batch in batch_generator(X_train.eval(), y_train.eval(), batch_size):
            values.node.backend.copy(X_batch, values.node.data)
            labels.node.backend.copy(y_batch, labels.node.data)
            loss.set_reeval()
            loss.eval()
            loss.zero_grad()
            W.grad().set_reeval()
            b.grad().set_reeval()
            optim.step()
        if (it % (iter//reports) == 0):
            print(f"{it} loss = ", loss.eval().item())
    
    logits_test = X_test @ W + b
    mse_test = mean_squared_error(logits_test, y_test)
    print(f'Test MSE: {mse_test.eval().item()}')

lr = 0.001
iter = 100
train(lr, iter, reports=10)