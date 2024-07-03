import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, load_digits
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import argparse

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', optimizer='sgd'):
        self.layers = len(hidden_sizes) + 1
        self.weights = []
        self.biases = []
        
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, len(layer_sizes)):
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2. / layer_sizes[i-1]))
            self.biases.append(np.zeros((1, layer_sizes[i])))
        
        self.activation = self.get_activation(activation)
        self.activation_prime = self.get_activation_prime(activation)
        self.optimizer = self.get_optimizer(optimizer)
        
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.v_biases = [np.zeros_like(b) for b in self.biases]
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases = [np.zeros_like(b) for b in self.biases]
    
    def get_activation(self, name):
        if name == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))
        elif name == 'relu':
            return lambda x: np.maximum(0, x)
        elif name == 'tanh':
            return lambda x: np.tanh(x)
        else:
            raise ValueError("Unsupported activation function")
    
    def get_activation_prime(self, name):
        if name == 'sigmoid':
            return lambda x: x * (1 - x)
        elif name == 'relu':
            return lambda x: (x > 0).astype(float)
        elif name == 'tanh':
            return lambda x: 1 - np.tanh(x)**2
        else:
            raise ValueError("Unsupported activation function")
    
    def get_optimizer(self, name):
        if name == 'sgd':
            return self.sgd
        elif name == 'adam':
            return self.adam
        else:
            raise ValueError("Unsupported optimizer")
    
    def forward(self, X, training=True):
        self.layer_outputs = [X]
        for i in range(self.layers):
            z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            if i < self.layers - 1:
                a = self.activation(z)
                if training:
                    a = self.dropout(a, 0.5)
            else:
                a = self.softmax(z)
            self.layer_outputs.append(a)
        return self.layer_outputs[-1]
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def dropout(self, x, keep_prob):
        mask = np.random.binomial(1, keep_prob, size=x.shape) / keep_prob
        return x * mask
    
    def backward(self, X, y, output):
        m = X.shape[0]
        deltas = [None] * self.layers
        gradients = {"weights": [], "biases": []}
        
        deltas[-1] = output - y
        for l in reversed(range(self.layers)):
            gradients["weights"].insert(0, np.dot(self.layer_outputs[l].T, deltas[l]) / m)
            gradients["biases"].insert(0, np.sum(deltas[l], axis=0, keepdims=True) / m)
            if l > 0:
                deltas[l-1] = np.dot(deltas[l], self.weights[l].T) * self.activation_prime(self.layer_outputs[l])
        
        return gradients
    
    def sgd(self, gradients, learning_rate):
        for l in range(self.layers):
            self.weights[l] -= learning_rate * gradients["weights"][l]
            self.biases[l] -= learning_rate * gradients["biases"][l]
    
    def adam(self, gradients, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        for l in range(self.layers):
            self.m_weights[l] = beta1 * self.m_weights[l] + (1 - beta1) * gradients["weights"][l]
            self.m_biases[l] = beta1 * self.m_biases[l] + (1 - beta1) * gradients["biases"][l]
            
            self.v_weights[l] = beta2 * self.v_weights[l] + (1 - beta2) * (gradients["weights"][l]**2)
            self.v_biases[l] = beta2 * self.v_biases[l] + (1 - beta2) * (gradients["biases"][l]**2)
            
            m_weights_hat = self.m_weights[l] / (1 - beta1)
            m_biases_hat = self.m_biases[l] / (1 - beta1)
            v_weights_hat = self.v_weights[l] / (1 - beta2)
            v_biases_hat = self.v_biases[l] / (1 - beta2)
            
            self.weights[l] -= learning_rate * m_weights_hat / (np.sqrt(v_weights_hat) + epsilon)
            self.biases[l] -= learning_rate * m_biases_hat / (np.sqrt(v_biases_hat) + epsilon)
    
    def train(self, X, y, learning_rate, epochs, batch_size=32, validation_data=None, early_stopping_patience=5):
        history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                output = self.forward(X_batch)
                gradients = self.backward(X_batch, y_batch, output)
                self.optimizer(gradients, learning_rate)
            
            train_loss, train_accuracy = self.evaluate(X, y)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            
            if validation_data:
                val_loss, val_accuracy = self.evaluate(*validation_data)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_accuracy)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}", end="")
                if validation_data:
                    print(f", Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
                else:
                    print()
        
        return history
    
    def predict(self, X):
        return self.forward(X, training=False)
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        loss = -np.mean(y * np.log(predictions + 1e-8))
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
        return loss, accuracy

def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.argmax(Z, axis=1).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=np.argmax(y, axis=1), alpha=0.8)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(np.arange(len(classes)) + 0.5, classes)
    plt.yticks(np.arange(len(classes)) + 0.5, classes)

def main(dataset, hidden_sizes, activation, optimizer, learning_rate, epochs, batch_size):
    if dataset == 'moons':
        X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
        y = np.eye(2)[y]
        classes = ['Class 0', 'Class 1']
    elif dataset == 'digits':
        digits = load_digits()
        X, y = digits.data, digits.target
        y = np.eye(10)[y]
        classes = [str(i) for i in range(10)]
    else:
        raise ValueError("Unsupported dataset")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_size = X.shape[1]
    output_size = y.shape[1]

    nn = NeuralNetwork(input_size, hidden_sizes, output_size, activation, optimizer)
    history = nn.train(X_train, y_train, learning_rate, epochs, batch_size, validation_data=(X_test, y_test))

    y_pred = nn.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)
    y_true_class = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true_class, y_pred_class)
    precision = precision_score(y_true_class, y_pred_class, average='weighted')
    recall = recall_score(y_true_class, y_pred_class, average='weighted')
    f1 = f1_score(y_true_class, y_pred_class, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_accuracy"], label="Train")
    plt.plot(history["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_history.png")

    if dataset == 'moons':
        plt.figure(figsize=(8, 6))
        plot_decision_boundary(X, y, nn)
        plt.savefig("decision_boundary.png")

    plot_confusion_matrix(y_true_class, y_pred_class, classes)
    plt.savefig("confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network from scratch")
    parser.add_argument("--dataset", type=str, default="moons", choices=["moons", "digits"], help="Dataset to use")
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[16, 8], help="Hidden layer sizes")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"], help="Activation function")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"], help="Optimizer")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    main(args.dataset, args.hidden_sizes, args.activation, args.optimizer, args.learning_rate, args.epochs, args.batch_size)