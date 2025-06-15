print("main_project.py запущен")

import os
import argparse
import ast
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import cv2
from torchvision.datasets import MNIST
from keras.utils import to_categorical


# -------------------- ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ ИЗ MNIST -------------------- #
def prepare_mnist_folders(base_folder='mnist_data'):
    structure = ['train', 'valid', 'test']
    for category in structure:
        for digit in range(10):
            os.makedirs(os.path.join(base_folder, category, str(digit)), exist_ok=True)


def convert_mnist_to_png(save_path='mnist_data'):
    prepare_mnist_folders(save_path)

    train_data = MNIST(root='data', train=True, download=True)
    test_data = MNIST(root='data', train=False, download=True)

    for i, (img, label) in enumerate(zip(train_data.data, train_data.targets)):
        split = 'train' if i < 50000 else 'valid'
        filename = os.path.join(save_path, split, str(label.item()), f"{i}.png")
        Image.fromarray(img.numpy(), mode='L').save(filename)

    for i, (img, label) in enumerate(zip(test_data.data, test_data.targets)):
        filename = os.path.join(save_path, 'test', str(label.item()), f"test_{i}.png")
        Image.fromarray(img.numpy(), mode='L').save(filename)


# -------------------- ПРЕОБРАЗОВАНИЕ ИЗОБРАЖЕНИЙ -------------------- #
def extract_features_from_directory(root_dir):
    samples, labels = [], []
    for digit in range(10):
        class_folder = os.path.join(root_dir, str(digit))
        if not os.path.isdir(class_folder):
            continue
        for file in os.listdir(class_folder):
            try:
                img = Image.open(os.path.join(class_folder, file)).convert('L').resize((28, 28), Image.LANCZOS)
                arr = np.array(img)
                if np.mean(arr) > 127:
                    arr = 255 - arr
                samples.append(arr.flatten() / 255.0)
                labels.append(to_categorical(digit, 10))
            except:
                continue
    return np.array(samples), np.array(labels)


def collect_dataset_sets(folder_path):
    train_x, train_y = extract_features_from_directory(os.path.join(folder_path, 'train'))
    test_x, test_y = extract_features_from_directory(os.path.join(folder_path, 'test'))
    valid_x, valid_y = extract_features_from_directory(os.path.join(folder_path, 'valid'))
    return train_x, train_y, test_x, test_y, valid_x, valid_y


# -------------------- ОБУЧЕНИЕ МОДЕЛИ -------------------- #
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def fit_model(x_train, y_train, x_val, y_val, layers_config, lr, n_epochs):
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    architecture = layers_config + [output_dim]

    np.random.seed(42)
    weights, biases = [], []
    prev_dim = input_dim

    for width in architecture:
        W = np.random.randn(prev_dim, width) * np.sqrt(2. / prev_dim)
        b = np.zeros(width)
        weights.append(W)
        biases.append(b)
        prev_dim = width

    for epoch in range(n_epochs):
        indices = np.random.permutation(len(x_train))
        x_shuffled, y_shuffled = x_train[indices], y_train[indices]

        activations = [x_shuffled]
        pre_acts = []

        for i in range(len(weights)):
            z = activations[-1] @ weights[i] + biases[i]
            pre_acts.append(z)
            a = relu(z) if i < len(weights) - 1 else softmax(z)
            activations.append(a)

        loss = -np.mean(y_shuffled * np.log(np.clip(activations[-1], 1e-8, 1.0)))
        grad = (activations[-1] - y_shuffled) / len(x_train)

        for i in reversed(range(len(weights))):
            if i < len(weights) - 1:
                grad *= relu_grad(pre_acts[i])
            dW = activations[i].T @ grad
            db = np.sum(grad, axis=0)
            weights[i] -= lr * dW
            biases[i] -= lr * db
            if i > 0:
                grad = grad @ weights[i].T

        train_acc = np.mean(np.argmax(activations[-1], axis=1) == np.argmax(y_shuffled, axis=1))
        val_pred = predict_model(x_val, weights, biases)
        val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))

        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    return weights, biases, len(weights)


# -------------------- ПРЕДСКАЗАНИЕ -------------------- #
def preprocess_image_file(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at {file_path}")
    img = cv2.resize(img, (28, 28)).astype('float32') / 255.0
    return img.reshape(1, -1)


def predict_model(data, W, b):
    output = data
    for i in range(len(W)):
        output = output @ W[i] + b[i]
        if i != len(W) - 1:
            output = relu(output)
    return softmax(output)


# -------------------- ИНТЕРФЕЙС -------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--probs', action='store_true')
    parser.add_argument('--weights_path', default='weights.npz')
    parser.add_argument('--path_to_dataset')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--hidden_size', type=str, default='[128, 64]')
    parser.add_argument('image', nargs='?')

    args = parser.parse_args()

    if args.train:
        if not args.path_to_dataset:
            parser.error('Необходимо указать путь к датасету (--path_to_dataset)')
        layers = ast.literal_eval(args.hidden_size)
        x_tr, y_tr, x_ts, y_ts, x_val, y_val = collect_dataset_sets(args.path_to_dataset)
        W, b, num = fit_model(x_tr, y_tr, x_val, y_val, layers, args.learning_rate, args.epochs)
        np.savez(args.weights_path, num, *W, *b)

        y_pred = predict_model(x_ts, W, b)
        y_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_ts, axis=1)

        print(f"Test accuracy: {np.mean(y_classes == y_true):.4f}")
        print(f"Test precision: {precision_score(y_true, y_classes, average='macro'):.4f}")
        print(f"Test recall: {recall_score(y_true, y_classes, average='macro'):.4f}")
        print(f"Test F1-score: {f1_score(y_true, y_classes, average='macro'):.4f}")
        print('Confusion Matrix:\n', confusion_matrix(y_true, y_classes))

    else:
        if not args.image:
            parser.error('Необходимо указать путь к изображению для предсказания')

        data = np.load(args.weights_path)
        L = data['arr_0']
        Ws = [data[f'arr_{i+1}'] for i in range(L)]
        Bs = [data[f'arr_{L+i+1}'] for i in range(L)]

        image = preprocess_image_file(args.image)
        result = predict_model(image, Ws, Bs)

        if args.probs:
            print("Class probabilities:")
            for i, prob in enumerate(result[0]):
                print(f"{i}: {prob:.4f}")
        else:
            print(f"Predicted digit: {np.argmax(result)}")

if __name__ == '__main__':
    main()
