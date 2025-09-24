import random
import numpy as np

# definimos clase para la función de costo Cross-Entropy
class CEC(object):

    @staticmethod
    def fn(a, y):
        """Devuelve el costo asociado entre la salida de la red 'a' y la salida deseada 'y'."""
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Devuelve el error delta de la capa de salida para cross-entropy."""
        return (a - y)


class Network(object):

    def __init__(self, sizes, cost=CEC, eta=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        sizes: lista con el número de neuronas en cada capa.
        eta: tasa de aprendizaje (Adam).
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.cost = cost

        # --- Parámetros para Adam ---
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0  # contador de iteraciones

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, test_data=None):
        """Entrena la red usando Adam."""
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        training_data = list(training_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch):
        """Actualiza pesos y sesgos usando backpropagation y Adam."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # --- Actualización con Adam ---
        self.t += 1
        lr_t = self.eta * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)

        for i in range(len(self.weights)):
            # actualizamos momentos de pesos
            grad_w = nabla_w[i] / len(mini_batch)
            self.m_w[i] = self.beta1*self.m_w[i] + (1-self.beta1)*grad_w
            self.v_w[i] = self.beta2*self.v_w[i] + (1-self.beta2)*(grad_w**2)

            m_w_hat = self.m_w[i] / (1 - self.beta1**self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2**self.t)
            self.weights[i] -= lr_t * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

            # actualizamos momentos de sesgos
            grad_b = nabla_b[i] / len(mini_batch)
            self.m_b[i] = self.beta1*self.m_b[i] + (1-self.beta1)*grad_b
            self.v_b[i] = self.beta2*self.v_b[i] + (1-self.beta2)*(grad_b**2)

            m_b_hat = self.m_b[i] / (1 - self.beta1**self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2**self.t)
            self.biases[i] -= lr_t * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def backprop(self, x, y):
        """Algoritmo de retropropagación que calcula los gradientes."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


#### Funciones auxiliares
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))