import numpy as np
import matplotlib.pyplot as plt

def test_model(model, ds):
    labels_test = []
    predictions_test = []
    for images, labels in ds:
        batch_predictions = model.predict(images)
        labels_test.extend(labels)
        predictions_test.extend(np.argmax(batch_predictions, axis=1))
    return labels_test, predictions_test


def plot_test_results(model, ds):
    fig = plt.figure(figsize=(15, 15))
    for images, labels in ds.take(1):
        y_pred = model.predict(images)
        y_pred_binary = np.argmax(y_pred, axis = 1)
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy())
            plt.title('label:{}, pred:{}'.format(int(labels[i]), y_pred_binary[i]))
            plt.axis("off")
    return fig