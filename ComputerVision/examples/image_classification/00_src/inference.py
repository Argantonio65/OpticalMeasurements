import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def get_img_ids(ds):
   file_names = [os.path.basename(name) for name in ds.file_paths]
   img_ids = np.array([line.split('_')[1].split('.')[0] for line in file_names])
   return img_ids


def link_waterbodies_coordinates(img_ids, y_pred, data_coord):
    data_coord['id'] = data_coord['id'].astype(str)
    df_img = pd.DataFrame({'id': img_ids.astype(str), 'prediction': y_pred})
    pred_coord = pd.merge(data_coord, df_img, on='id')
    return pred_coord


def save_prediction_geojson(pred_coord, dir_data, filename_geojson):
   full_path_pred_coordinates = os.path.join(dir_data, f'{filename_geojson}_predicted.geojson')
   pred_coord.to_file(full_path_pred_coordinates, driver='GeoJSON')
   return print(f'Your predicitons linked to the coordinates of the waterbodies is saved as: {full_path_pred_coordinates}')


def plot_inference_results(model, ds):
    fig = plt.figure(figsize=(15, 15))
    for images in ds.take(1):
      y_pred = model.predict(images)
      y_pred_binary = np.argmax(y_pred, axis = 1)
      for i in range(20):
          plt.subplot(5, 4, i + 1)
          plt.imshow(images[i].numpy())
          plt.title('pred:{}'.format(y_pred_binary[i]))
          plt.axis("off")
    return fig