import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

def load_label_names():
  meta = unpickle(os.path.join('cifar-10', 'batches.meta'))
  label_names = meta[b'label_names']
  return [label.decode('utf-8') for label in label_names]

label_names = load_label_names()

def load_data():
  data = []
  labels = []
  for i in range(1, 6):
    batch = unpickle(os.path.join('cifar-10', 'data_batch_' + str(i)))
    for img, label in zip(batch[b'data'], batch[b'labels']):
      data.append(img)
      labels.append(label)
  data = np.array(data)
  labels = np.array(labels)
  return data, labels

data, labels = load_data()

# Display the first 10 images
for i in range(10):
  image = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
  plt.figure()
  plt.imshow(image)
  plt.title(f"Label: {label_names[labels[i]]}")
  plt.show()
