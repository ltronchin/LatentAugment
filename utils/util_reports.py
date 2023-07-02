import base64
import os
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure, show, output_file
from bokeh.transform import factor_mark, factor_cmap

from utils import util_path

plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 16

def get_cmap(n: int, name: str ='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)

def plot_training(history, plot_training_dir, columns_to_plot=None, **plot_args):
    util_path.create_dir(plot_training_dir)

    if not isinstance(history, pd.DataFrame):
        history = pd.DataFrame.from_dict(history, orient='index').transpose()

    columns_in_history = list(history.keys())
    if columns_to_plot is not None:
        columns_to_plot = intersection(columns_in_history, columns_to_plot)
    else:
        columns_to_plot = columns_in_history

    cmap = get_cmap(n=len(columns_to_plot)+1, name='hsv')
    fig=plt.figure(figsize=(8, 6))
    for idx, key in enumerate(columns_to_plot):
        plt.plot(history[key], label=key, c=cmap(idx))

    plt.title(plot_args['title'])
    plt.xlabel(plot_args['xlab'])
    plt.ylabel(plot_args['ylab'])
    plt.legend()
    fig.savefig(os.path.join(plot_training_dir, f"{plot_args['img_name']}.png"), dpi=400, format='png')
    #plt.show()

def show_activation(x, layer, report_dir):
    assert isinstance(x, torch.Tensor)

    report_dir = os.path.join(report_dir, 'activations')
    util_path.create_dir(report_dir)

    xgrid = torch.permute(x, (1, 0, 2, 3)).detach().cpu()
    nrow = int(np.sqrt(xgrid.shape[0]))
    tot = nrow * nrow

    grid_img_args={
        'normalize': True,
        'value_range': (-1,1),
        }
    torchvision.utils.save_image(xgrid[:tot], os.path.join(report_dir, f"activation_grid_{layer}.png"), **grid_img_args)

    #fig = plt.figure(figsize=(8, 6))
    #grid_img = torchvision.utils.make_grid(xgrid[:tot], nrow=nrow, normalize=True, )
    #plt.imshow(grid_img.permute(1, 2, 0), cmap='gray')
    #plt.axis('off')
    #fig.savefig(os.path.join(report_dir, f"activation_grid_{layer}.png"), format='png')
    #plt.show()

def scatter_plot(output_dir, data, label, output_name='umap_plot', labels_name=None, colors=None, markers=None, sizes=None, alphas=None):
    # Plot the scatter plot.
    if labels_name is None:
        labels_name = ['Real data', 'LatentAugment', 'Standard SG2 DA']
    if alphas is None:
        alphas = [0.8, 0.5, 0.8]
    if sizes is None:
        sizes = [50, 50, 50]
    if markers is None:
        markers = ['*', 'o', '^']
    if colors is None:
        colors = ['blue', 'limegreen', 'lightgray']

    fig, ax = plt.subplots()
    # Plot each label separately with the appropriate color, marker, size, and alpha.
    for i in range(len(np.unique(label))):
        mask = (label == i)
        ax.scatter(
            data[mask, 0], data[mask, 1], c=colors[i], edgecolor='none', marker=markers[i], s=sizes[i], alpha=alphas[i],
            label=labels_name[i]
        )
    # Add axis labels and title.
    plt.xlabel('Embedding 1')
    plt.ylabel('Embedding 2')
    plt.legend()

    fig.savefig(os.path.join(output_dir, f"{output_name}.png"), dpi=400, format='png')
    # Show the plot.
    plt.show()

def embeddable_image(imgs_data):

    if imgs_data.min() < -1 or imgs_data.max() > 1:
        imgs_data = np.clip(imgs_data, -1.0, 1.0)

    img_data = ((imgs_data + 1) * 255 / 2).astype(np.uint8)
    image = Image.fromarray(img_data,
                            mode='L')  # image = Image.fromarray(img_data, mode='L').resize((64, 64), Image.Resampling.BICUBIC)
    buffer = BytesIO()
    image.save(buffer, format='png')
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()

def scatter_plot_interactive(output_dir, data, label, imgs, output_name):
    label_map = {0: 'Real data', 1: 'LatentAugment', 2: 'Standard SG2 DA'}
    imgs_df = pd.DataFrame(data, columns=('x', 'y'))
    imgs_df['aug'] = [label_map[y] for y in label]
    imgs_df['image'] = list(map(embeddable_image, imgs))

    output_file(os.path.join(output_dir, f'{output_name}.html'))

    AUGMENTATIONS = ['Real data', 'LatentAugment', 'Standard SG2 DA']
    MARKERS = ['star', 'circle', 'triangle']
    COLORS = ['blue', 'limegreen', 'lightgray']

    datasource = ColumnDataSource(imgs_df)

    plot_figure = figure(
        title='UMAP projection',
        outer_width=1200,
        outer_height=1200,
        x_range=(5, 11),
        y_range=(-1, 5),
        tools='pan, wheel_zoom, reset'
    )
    plot_figure.xaxis.axis_label = 'Embedding 1'
    plot_figure.yaxis.axis_label = 'Embedding 2'

    plot_figure.add_tools(HoverTool(tooltips="""
    <div>
        <div>
            <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
        </div>
        <div>
            <span style='font-size: 16px; color: #224499'></span>
            <span style='font-size: 18px'>@aug</span>
        </div>
    </div>
    """))

    plot_figure.scatter(
        "x",
        "y",
        source=datasource,
        legend_group="aug",
        fill_alpha=0.4,
        size=12,
        marker=factor_mark('aug', MARKERS, AUGMENTATIONS),
        color=factor_cmap('aug', COLORS, AUGMENTATIONS)
    )
    plot_figure.legend.location = "top_left"
    plot_figure.legend.title = "Augmentations"
    show(plot_figure)

def dump_images(output_dir, imgs, fname):
    for i, x in enumerate(imgs):
        if x.min() < -1 or x.max() > 1:
            x = np.clip(x, -1.0, 1.0)
        x = ((x + 1) / 2 * 255.0).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"{fname}_{i}.png"), x)