import itertools

from numpy import argmax, argmin, arange

import matplotlib.pyplot as plt
from plotly.express import histogram
from plotly.offline import offline
from plotly.graph_objs import Figure
from matplotlib.animation import FuncAnimation

from config import Path


def plot_number_of_images(dict_folder, data_type="train"):
    if data_type not in [subfolder.rstrip("/") for subfolder in Path.SUBFOLDERS]:
        print("[E]: Entered Data Type is not valid. Please use 'train', 'test' or 'val'.")
        return None

    df = dict_folder[data_type]
    
    fig = histogram(data_frame=df,
                    y=df['label'],
                    template='plotly_dark',
                    color=df['label'].values,
                    title=f'Number of images in each class of the {data_type} data.')
    
    return fig

def save_plot(obj, output_path="Output.png"):
    
    # For plt.Figure object use savefig method
    if isinstance(obj, plt.Figure):
        # Check for file format
        if output_path.lower().endswith(('.jpg', '.png', '.jpeg')):    
            print(f"[I] Saving plot to {Path.PLOTS+output_path}.")
            obj.savefig(Path.PLOTS+output_path, bbox_inches='tight')
        else:
            raise ValueError("[E] File ending does not match plotting type. Please use jpg, png or jpeg file format for static figure.")
    
    # For animaton object use save method
    elif isinstance(obj, FuncAnimation):
        # Check for file format
        if output_path.lower().endswith('.gif'):    
            print(f"[I] Saving plot to {Path.PLOTS+output_path}.")
            obj.save(Path.PLOTS+output_path, writer='pillow')
        else:
            raise ValueError("[E] File ending does not match plotting type. Please use gif file format for animation plot.")
        
    # For plotly type
    elif isinstance(obj, Figure):
        if output_path.lower().endswith('.html'):
            print(f"[I] Saving plot to {Path.PLOTS+output_path}.")
            offline.plot(obj, filename=Path.PLOTS+output_path, auto_open=False)
        else:
            raise ValueError("[E] File ending does not match plotting type. Please use html file format for plotly image.")

    else:
        raise ValueError("[E] Unsupported object type. Please use fig, plotly Figure or animation object.")


def plot_train_xrays(train_gen):
    gen_dict = train_gen.class_indices
    classes = list(gen_dict.keys())
    images, labels = next(train_gen)

    fig = plt.figure(figsize=(20, 20))
    
    for i in range(16):
        plt.subplot(4, 4, i+1)
        image = images[i] / 255
        
        plt.imshow(image)
        class_index = argmax(labels[i])
        class_name = classes[class_index]
        plt.title(class_name, color='blue', fontsize=12)
        plt.axis('off')

    return fig


def plot_model_accuracy(train_loss, train_acc, val_acc, val_loss):

    val_lowest = val_loss[argmax(val_loss)]
    val_highest = val_acc[argmax(val_acc)]

    epochs = [i+1 for i in range(len(train_acc))]

    index_loss = argmin(val_loss)
    index_acc = argmax(val_acc)

    loss_label = f'Best Epochs = {str(index_loss+1)}.'
    acc_label = f'Best Epochs = {str(index_acc+1)}.'

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    plt.style.use('fivethirtyeight')

    axes[0].plot(epochs, train_loss, 'r', label='Training Loss')
    axes[0].plot(epochs, val_loss, 'g', label='Validation Loss')
    axes[0].scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(epochs, train_acc, 'r', label='Training Accuracy')
    axes[1].plot(epochs, val_acc, 'g', label='Validation Accuracy')
    axes[1].scatter(index_acc + 1, val_highest, s=150, c='blue', label=acc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()

    return fig

def plot_confusion_matrix(cm, classes):
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    return fig