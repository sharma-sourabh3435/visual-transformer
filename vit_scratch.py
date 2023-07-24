import numpy as np
import matplotlib.pyplot as plt
import pickle
import math


def load_cifar100():
    with open('cifar-100-python/train', 'rb') as f:
        train_data = pickle.load(f, encoding='latin1')

    with open('cifar-100-python/test', 'rb') as f:
        test_data = pickle.load(f, encoding='latin1')

    return train_data, test_data


def preprocess_cifar100(data):
    images = data['data'].reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
    labels = np.array(data['fine_labels'])

    return images, labels


def normalize_data(images):
    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))

    images = (images - mean) / std

    return images


def split_data(images, labels, train_ratio=0.8):
    num_train = int(train_ratio * len(images))
    train_images = images[:num_train]
    train_labels = labels[:num_train]
    test_images = images[num_train:]
    test_labels = labels[num_train:]

    return train_images, train_labels, test_images, test_labels


class ViT:
    def __init__(self, input_shape, patch_size, num_patches, hidden_size, num_classes, num_heads, num_layers, mlp_size,
                 dropout_rate):
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_size = mlp_size
        self.dropout_rate = dropout_rate
        self.weights = {}
        self.biases = {}

    def initialize_weights(self):
        std = 1 / math.sqrt(self.hidden_size)
        self.weights['W_q'] = np.random.normal(0, std, (self.hidden_size, self.hidden_size))
        self.weights['W_k'] = np.random.normal(0, std, (self.hidden_size, self.hidden_size))
        self.weights['W_v'] = np.random.normal(0, std, (self.hidden_size, self.hidden_size))
        self.weights['W_o'] = np.random.normal(0, std, (self.hidden_size, self.hidden_size))
        self.weights['W_mlp1'] = np.random.normal(0, std, (self.hidden_size, self.mlp_size))
        self.weights['W_mlp2'] = np.random.normal(0, std, (self.mlp_size, self.hidden_size))
        self.weights['W_cls'] = np.random.normal(0, std, (self.hidden_size, self.num_classes))
        self.biases['b_cls'] = np.zeros(self.num_classes)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def apply_mlp(self, x):
        return np.maximum(0, np.dot(x, self.weights['W_mlp1'])) @ self.weights['W_mlp2']

    def apply_attention(self, x, mask):
        q = np.dot(x, self.weights['W_q'])
        k = np.dot(x, self.weights['W_k'])
        v = np.dot(x, self.weights['W_v'])

        qk = np.dot(q, k.T) / np.sqrt(self.hidden_size)
        qk += (mask * -1e9)
        attn_weights = self.softmax(qk)

        return np.dot(attn_weights, v)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1, self.hidden_size))

        for _ in range(self.num_layers):
            x_res = x

            x = self.apply_attention(x, mask)
            x += x_res

            x_res = x

            x = self.apply_mlp(x)
            x += x_res

        x = x[:, 0]  # Extract the first token representing the entire image
        x = np.dot(x, self.weights['W_cls']) + self.biases['b_cls']

        return x

    def backward(self, x, y_true, grad):
        batch_size = x.shape[0]

        # Backpropagate through the linear classifier
        dW_cls = np.dot(x.T, grad)
        db_cls = np.sum(grad, axis=0)

        # Backpropagate through the MLP layer
        grad = np.dot(grad, self.weights['W_cls'].T)
        grad = grad * (x > 0)
        dW_mlp2 = np.dot(x.T, grad)
        dW_mlp1 = np.dot(x.T.reshape((batch_size, -1, self.hidden_size)), grad)
        dW_mlp1 = np.sum(dW_mlp1, axis=0)

        # Backpropagate through the attention and mlp blocks (for num_layers times)
        for _ in range(self.num_layers):
            grad = grad + np.dot(grad, self.weights['W_mlp2'].T)
            grad = grad * (x > 0)
            dW_mlp2 = np.dot(x.T, grad) + dW_mlp2
            dW_mlp1 = np.dot(x.T.reshape((batch_size, -1, self.hidden_size)), grad) + dW_mlp1

            # Attention Block
            grad = grad + np.dot(grad, self.weights['W_v'].T)
            grad = grad * (x > 0)
            dW_v = np.dot(x.T, grad)
            dW_k = np.dot(x.T, grad)
            dW_q = np.dot(x.T, grad)

            grad = np.dot(grad, self.weights['W_k'].T) + np.dot(grad, self.weights['W_q'].T)
            grad = grad * (x > 0)
            dW_k += np.dot(x.T, grad)
            dW_q += np.dot(x.T, grad)

            grad = grad * (x > 0)
            dW_q = np.dot(x.T, grad)

        # Update the weights using gradient descent
        self.weights['W_cls'] -= learning_rate * dW_cls / batch_size
        self.biases['b_cls'] -= learning_rate * db_cls / batch_size
        self.weights['W_mlp2'] -= learning_rate * dW_mlp2 / batch_size
        self.weights['W_mlp1'] -= learning_rate * dW_mlp1 / batch_size
        self.weights['W_v'] -= learning_rate * dW_v / batch_size
        self.weights['W_k'] -= learning_rate * dW_k / batch_size
        self.weights['W_q'] -= learning_rate * dW_q / batch_size

    def train(self, train_images, train_labels, num_epochs, learning_rate, batch_size):
        num_batches = len(train_images) // batch_size

        for epoch in range(1, num_epochs + 1):
            indices = np.random.permutation(len(train_images))
            train_images = train_images[indices]
            train_labels = train_labels[indices]

            for i in range(num_batches):
                batch_images = train_images[i * batch_size: (i + 1) * batch_size]
                batch_labels = train_labels[i * batch_size: (i + 1) * batch_size]

                with open('progress.txt', 'a') as f:
                    f.write(f'Epoch: {epoch}/{num_epochs}, Batch: {i + 1}/{num_batches}\n')

                # Forward pass
                output = self.forward(batch_images)

                # Compute the loss (you'll need to define a suitable loss function)
                loss = compute_loss(output, batch_labels)

                # Compute the gradient of the loss with respect to the output
                grad = compute_gradient(output, batch_labels)  # Implement this function

                # Backward pass
                self.backward(batch_images, batch_labels, grad)

    def cross_entropy_loss(y_pred, y_true):
        # Compute the softmax of the predictions
        softmax_preds = np.exp(y_pred - np.max(y_pred, axis=-1, keepdims=True))
        softmax_preds /= np.sum(softmax_preds, axis=-1, keepdims=True)

        # Compute the cross-entropy loss
        num_samples = y_pred.shape[0]
        loss = -np.sum(np.log(softmax_preds[np.arange(num_samples), y_true])) / num_samples

        return loss

    def compute_gradient(y_pred, y_true):
        num_samples = y_pred.shape[0]

        # Compute the softmax of the predictions
        softmax_preds = np.exp(y_pred - np.max(y_pred, axis=-1, keepdims=True))
        softmax_preds /= np.sum(softmax_preds, axis=-1, keepdims=True)

        # Compute the gradient of the cross-entropy loss with respect to the predictions
        grad = softmax_preds
        grad[np.arange(num_samples), y_true] -= 1
        grad /= num_samples

        return grad

    def test(self, test_images, test_labels):
        num_samples = len(test_images)
        num_correct = 0

        for i in range(num_samples):
            image = test_images[i]
            label = test_labels[i]

            predicted_label = np.argmax(self.forward(image[np.newaxis, ...]))
            if predicted_label == label:
                num_correct += 1

        accuracy = num_correct / num_samples
        print(f'Test Accuracy: {accuracy * 100:.2f}%')


train_data, test_data = load_cifar100()
train_images, train_labels = preprocess_cifar100(train_data)
test_images, test_labels = preprocess_cifar100(test_data)

train_images = normalize_data(train_images)
test_images = normalize_data(test_images)


model = ViT(
    input_shape=(32, 32, 3),
    patch_size=8,
    num_patches=16,
    hidden_size=256,
    num_classes=100,
    num_heads=4,
    num_layers=2,
    mlp_size=512,
    dropout_rate=0.1
)
model.initialize_weights()


model.train(train_images, train_labels, num_epochs=10, learning_rate=0.001, batch_size=64)
model.test(test_images, test_labels)
