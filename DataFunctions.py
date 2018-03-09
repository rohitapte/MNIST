import numpy as np
import gzip
import matplotlib.pyplot as plt

def _read32(bytestream):
    dt=np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting',filename)
    with gzip.open(filename) as bytestream:
        magic=_read32(bytestream)
        if magic!=2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %(magic,filename))
        num_images=_read32(bytestream)
        rows=_read32(bytestream)
        cols=_read32(bytestream)
        buf=bytestream.read(rows * cols * num_images)
        data=np.frombuffer(buf,dtype=np.uint8)
        data=data.reshape(num_images,rows,cols,1)
    return data

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def extract_labels(filename, one_hot=False):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %(magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
    return labels

def plotData(imageData):
    plt.gray()
    plt.imshow(np.squeeze(imageData, axis=2))

TRAIN_IMAGES = 'data/train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'data/train-labels-idx1-ubyte.gz'
TEST_IMAGES = 'data/t10k-images-idx3-ubyte.gz'
TEST_LABELS = 'data/t10k-labels-idx1-ubyte.gz'
VALIDATION_SIZE = 5000
train_images=extract_images(TRAIN_IMAGES)
train_labels=extract_labels(TRAIN_LABELS)