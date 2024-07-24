import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def extract_labels(dataset):
    labels = []
    for _, label in tqdm(dataset):
        labels.append(label.numpy())
    return np.concatenate(labels, axis=0)

def build_training_dataset(dir_data, subset, image_size, batch_size):
  ds = tf.keras.preprocessing.image_dataset_from_directory( #write: expects data in specific format (folder names)
      dir_data,
      validation_split=.20,
      subset=subset,
      label_mode="categorical",
      seed=123,
      image_size=image_size,
      batch_size=1)

  # Extract labels from the training dataset (useful to created balancing weights)
  labels = extract_labels(ds)

  ds_class_names = tuple(ds.class_names)
  ds_size = ds.cardinality().numpy()
  ds = ds.unbatch().batch(batch_size)
  if subset == "training":
    ds = ds.repeat()
  return ds, ds_class_names, ds_size, labels
  

def augment_data(do_data_augmentation, preprocessing_model):
  if do_data_augmentation:
    preprocessing_model.add(
        tf.keras.layers.RandomRotation(40))
    preprocessing_model.add(
        tf.keras.layers.RandomZoom(0.2, 0.2))
    preprocessing_model.add(
        tf.keras.layers.RandomFlip(mode="horizontal"))
    preprocessing_model.add(
        tf.keras.layers.RandomFlip(mode="vertical"))
  return preprocessing_model


def plot_training_data(ds, number_of_images):
  ncols = 4
  nrows = number_of_images // ncols + 1
  fig = plt.figure(figsize=(15, 15))
  for images, labels in ds.take(1):
    for i in range(number_of_images):
      plt.subplot(nrows, ncols, i + 1)
      plt.imshow(images[i].numpy())
      plt.title(np.argmax(labels[i].numpy()))
      plt.axis("off")
  return fig
