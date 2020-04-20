import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import pandas as pd
from misc_funcs import FULL_EM
import seaborn as sns



COLOR_PALETTE = ["#f9ed69", "#f08a5d", "#b83b5e", "#00adb5", "#6a2c70", "#393e46"]


def plot_metric(ax, metrics, metric_title):
    x_data = range(len(metrics))
    ax.plot(x_data, metrics[:, 0], color=COLOR_PALETTE[1], label="train")
    ax.plot(x_data, metrics[:, 1], color=COLOR_PALETTE[3], label="validation")
    ax.set_title("{} progression".format(metric_title))
    ax.set_xlabel("epochs")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(metric_title)
    ax.legend()
    
    
def plot_metrics(history):
    """
    plots the accuracy and loss of the history object
    Args:
        history(list): list of dictionaries (one per epoch) containing the metrics of interest
    """
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    accs = np.array([[ep["accuracy_score"], ep["valid_acc"]] for ep in history])
    plot_metric(ax[0], accs, "Accuracy")
    losses = np.array([[ep["train_loss"], ep["valid_loss"]] for ep in history])
    plot_metric(ax[1], losses, "Cross entropy loss")
    fig.savefig("regularized.png")
    
def vis_sample(sample,title=None):
     """
    visualize mfcc samples
    Args:
        sample(np.array): mfcc coefficient of a wav file to visualize as a single layered image
    """
    fig, ax = plt.subplots(figsize=(15, 15))
    plt.imshow(np.transpose(sample), cmap="gray")
    if title is not None:
        fig.savefig('results/'+title+'.png')
    
def convert_history(hist):
    """
    converts a tensorflow history to workable format
    Args:
        hist(tf.history): a tensorflow history object
    """
    ret = []
    for val_loss,val_acc,loss,acc in zip(hist['val_loss'],
                                        hist['val_accuracy'],
                                        hist['loss'],
                                        hist['accuracy']):
        app = {}
        app['accuracy_score'] = acc
        app['valid_acc'] = val_acc
        app['train_loss'] = loss
        app['valid_loss'] = val_loss
        ret.append(app)
    return ret

def plot_confusion_matrix(matrix):
    """
    plots the confusion matrix
    Args:
        matrix(np.array): confusion matrix
    """
    df_cm = pd.DataFrame(matrix, index = FULL_EM.values(),
                      columns = FULL_EM.values())
    fig,ax = plt.subplots(1,figsize = (10,10))
    sns.heatmap(df_cm, annot=True,ax=ax)
    fig.savefig("confusion.png")