from config import PATH_SUB, PATH_PLOT_FOLDER
from plotly.express import histogram
from plotly.io import write_image
from plotly.graph_objs import Figure
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

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
            raise ValueError("File ending does not match plotting type. Please use jpg, png or jpeg file format.")
    
    # For animaton object use save method
    elif isinstance(obj, FuncAnimation):
        # Check for file format
        if output_path.lower().endswith('.gif'):    
            print(f"Saving plot to {PATH_PLOT_FOLDER+output_path}.")
            obj.save(PATH_PLOT_FOLDER+output_path, writer='pillow')
        else:
            raise ValueError("File ending does not match plotting type. Please use gif file format.")
        
    # For plotly type
    elif isinstance(obj, Figure):
        if output_path.lower().endswith(('.jpg', '.png', '.jpeg')):
            print(f"Saving plot to {PATH_PLOT_FOLDER+output_path}.")
            write_image(obj, PATH_PLOT_FOLDER+output_path)

    else:
        raise ValueError("Unsupported object type. Please use fig, plotly Figure or animation object.")

