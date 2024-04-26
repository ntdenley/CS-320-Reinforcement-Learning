# downloads the mnist dataset, created by chatgpt
import os
import gzip
import pandas as pd
import numpy as np
import ml.array as ml
from urllib.request import urlretrieve

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    for filename in files:
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urlretrieve(base_url + filename, filename)
            print(f"{filename} downloaded!")

def extract_images_and_labels(image_filename, label_filename):
    with gzip.open(image_filename, 'rb') as f:
        # Skip the magic number and dimension info
        f.read(16)
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784)
    
    with gzip.open(label_filename, 'rb') as f:
        # Skip the magic number
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    return pd.DataFrame(data=images, dtype=np.uint8), pd.Series(labels)

def save_to_csv(dataframe, filename):
    dataframe.to_csv(filename, index=False)
    print(f"{filename} has been saved.")

def load_mnist():
    download_mnist()
    train_images, train_labels = extract_images_and_labels("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
    test_images, test_labels = extract_images_and_labels("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
    
    train_images['label'] = train_labels
    test_images['label'] = test_labels
    
    save_to_csv(train_images, "mnist_train.csv")
    save_to_csv(test_images, "mnist_test.csv")
    
    # Load the CSVs into pandas DataFrame
    df_train = pd.read_csv("mnist_train.csv")
    df_test = pd.read_csv("mnist_test.csv")
    
    print("DataFrames loaded with MNIST data.")
    
    # Extract labels and convert them to numpy array
    train_labels = ml.Array(df_train['label'].to_numpy().tolist())
    test_labels = ml.Array(df_test['label'].to_numpy().tolist())

    # Drop the label column to isolate the image data
    train_images = ml.Array(df_train.drop('label', axis=1).to_numpy().tolist())
    test_imageget_data = ml.Array(df_test.drop('label', axis=1).to_numpy().tolist())

    print("Arrays loaded with MNIST data.")

    return train_images, train_labels, test_images, test_labels