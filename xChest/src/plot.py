from config import PATH_SUB, PATH_PLOT_FOLDER
from plotly.express import histogram
from plotly.offline import offline
from plotly.graph_objs import Figure
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from numpy import argmax
from sklearn.preprocessing import MinMaxScaler

def plot_number_of_images(dict_folder, data_type="train"):
    if data_type not in [subfolder.rstrip("/") for subfolder in PATH_SUB]:
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
            print(f"Saving plot to {PATH_PLOT_FOLDER+output_path}.")
            obj.savefig(PATH_PLOT_FOLDER+output_path, bbox_inches='tight')
        else:
            raise ValueError("File ending does not match plotting type. Please use jpg, png or jpeg file format for static figure.")
    
    # For animaton object use save method
    elif isinstance(obj, FuncAnimation):
        # Check for file format
        if output_path.lower().endswith('.gif'):    
            print(f"Saving plot to {PATH_PLOT_FOLDER+output_path}.")
            obj.save(PATH_PLOT_FOLDER+output_path, writer='pillow')
        else:
            raise ValueError("File ending does not match plotting type. Please use gif file format for animation plot.")
        
    # For plotly type
    elif isinstance(obj, Figure):
        if output_path.lower().endswith('.html'):
            print(f"Saving plot to {PATH_PLOT_FOLDER+output_path}.")
            offline.plot(obj, filename=PATH_PLOT_FOLDER+output_path, auto_open=False)
        else:
            raise ValueError("File ending does not match plotting type. Please use html file format for plotly image.")

    else:
        raise ValueError("Unsupported object type. Please use fig, plotly Figure or animation object.")


def plot_train_xrays(train_gen):
    gen_dict = train_gen.class_indices
    classes = list(gen_dict.keys())
    images, labels = next(train_gen)

    fig = plt.figure(figsize=(20, 20))
    
    for i in range(16):
        plt.subplot(4, 4, i+1)
        image = images[i] / 255
        
        # Flatten the image
        #flattened_image = image.reshape(-1)
        
        # Scale the flattened image
        #scaled_image = MinMaxScaler().fit_transform(flattened_image.reshape(-1, 1)).reshape(image.shape)
        
        plt.imshow(image)
        class_index = argmax(labels[i])
        class_name = classes[class_index]
        plt.title(class_name, color='blue', fontsize=12)
        plt.axis('off')

    return fig