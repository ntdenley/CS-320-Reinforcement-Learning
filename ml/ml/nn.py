from .array import Array, rand
from math import sqrt

''' optimizers '''
class SGD:
    def __init__(self, lr, params):
        self.backend = Array([1]).node.backend
        self.lr = lr
        self.params = params
        self.temp_arrays = [self.backend.empty(p.node.shape) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            t = self.temp_arrays[i]
            p.grad().set_reeval()
            self.backend.multiply(self.lr, p.grad().eval(), t)
            self.backend.negative(t, t)
            self.backend.add(p.node.data, t, p.node.data)

# made with help from chatgpt
class Adam:
    def __init__(self, lr, params, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.params = params
        self.t = 0  # Time step
        self.backend = Array([1]).node.backend  # The NumpyBackend instance

        # Initialize moment vectors
        self.m = [self.backend.empty(p.node.shape) for p in params]
        self.v = [self.backend.empty(p.node.shape) for p in params]

        # Initialize moments to zero
        for i in range(len(self.params)):
            self.backend.fill_inplace(self.m[i], 0)
            self.backend.fill_inplace(self.v[i], 0)

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            grad = p.grad().eval()
            
            # Update biased first moment estimate
            self.backend.multiply(self.beta1, self.m[i], self.m[i])  # in-place scale by beta1
            self.backend.multiply((1 - self.beta1), grad, grad)  # scale grad by (1-beta1)
            self.backend.add(self.m[i], grad, self.m[i])  # in-place update of first moment

            # Update biased second moment estimate
            self.backend.multiply(self.beta2, self.v[i], self.v[i])  # in-place scale by beta2
            self.backend.square(grad, grad)  # square the gradients in-place
            self.backend.multiply((1 - self.beta2), grad, grad)  # scale squared grad by (1-beta2)
            self.backend.add(self.v[i], grad, self.v[i])  # in-place update of second moment

            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            self.backend.sqrt(v_hat, v_hat)  # in-place square root
            self.backend.add(v_hat, self.epsilon, v_hat)  # in-place addition of epsilon
            self.backend.reciprocal(v_hat, v_hat)  # in-place reciprocal
            self.backend.multiply(m_hat, v_hat, v_hat)  # in-place multiplication with m_hat
            self.backend.multiply(self.lr, v_hat, v_hat)  # in-place multiplication with learning rate
            self.backend.add(p.node.data, -v_hat, p.node.data)  # update parameters

''' some functions '''
def one_hot_encode(data, classes):
    out = []
    for d in data.eval():
        one_hot = [0 for _ in range(classes)]
        one_hot[int(d)] = 1
        out.append(one_hot)
    return Array(out)

def softmax_cross_entropy_loss(logits, targets):
    loss = -(targets * logits.softmax(1).log()).sum() / logits.node.shape[0]
    return loss
