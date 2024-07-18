import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_training_results(hist):
    fig, axes = plt.subplots(1, 2, figsize=(10,10))
    for i, x in enumerate(['loss', 'accuracy']):
        axes[i].set_ylabel(f'{x} (training and validation)')
        axes[i].set_xlabel("Training steps")
        axes[i].plot(hist[x])
        axes[i].plot(hist[f'val_{x}'])
    axes[0].set_ylim([0,1])
    axes[1].set_ylim([0,2])
    return fig


def save_model(model, dir_model, name_model, hist):
    full_path_model = os.path.join(dir_model, f'{name_model}_finetuned.h5')
    model.save(full_path_model)
    hist_df = pd.Dataframe(hist)
    hist_df.to_csv(os.path.join(dir_model, f'{name_model}_finetuned.csv'))
    return  print(f'Your fine-tuned model is saved as: {full_path_model}')

