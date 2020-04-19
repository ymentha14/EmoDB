import matplotlib.pyplot as plt

COLOR_PALETTE = ["#f9ed69", "#f08a5d", "#b83b5e", "#00adb5", "#6a2c70", "#393e46"]


def plot_metric(ax, metrics, metric_title):
    x_data = range(len(metrics))
    ax.plot(x_data, metrics[:, 0], color=COLOR_PALETTE[1], label="train")
    ax.plot(x_data, metrics[:, 1], color=COLOR_PALETTE[3], label="validation")
    ax.set_title("{} progression of best model".format(metric_title))
    ax.set_xlabel("epochs")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(metric_title)
    ax.legend()
    
    
def plot_metrics(history):
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    accs = np.array([[ep["accuracy_score"], ep["valid_acc"]] for ep in net.history_])
    plot_metric(ax[0], accs, "Accuracy")
    losses = np.array([[ep["train_loss"], ep["valid_loss"]] for ep in net.history_])
    plot_metric(ax[1], losses, "Cross entropy loss")
    
# visualize some random speeches
def vis_sample(sample):
    fig, ax = plt.subplots(figsize=(10, 15))
    plt.imshow(np.transpose(sample), cmap="gray")