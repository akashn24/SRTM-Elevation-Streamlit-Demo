import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.colors import LightSource

def plottable_3d_info(df: pd.DataFrame):
    """
    Transform Pandas data into a format that's compatible with
    Matplotlib's surface and wireframe plotting.
    """
    index = df.index
    columns = df.columns
    
    y_label = df.index.to_series().apply(lambda x: np.round(x,2))
    x_label = df.columns.to_series().apply(lambda x: np.round(x,2))

    x, y = np.meshgrid(np.arange(len(columns)), np.arange(len(index)))
    z = np.array([[df[c][i] for c in columns] for i in index])
    
    xticks = dict(ticks=np.arange(len(columns)), labels=x_label)
    yticks = dict(ticks=np.arange(len(index)), labels=y_label)
    
    return x, y, z, xticks, yticks


def surface_plot(df_to_plot):
    ### Transform data from dataframe to Matplotlib friendly format.
    x, y, z, xticks, yticks = plottable_3d_info(df_to_plot)

    # Set up axes and put data on the surface.
    fig, axes = plt.subplots(subplot_kw=dict(projection='3d'))
    
    # Specifies colours of surface plot
    ls = LightSource(270, 45)
    rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')

    axes.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb, linewidth=0, antialiased=False, shade=False)

    axes.set_xlabel('Latitude')
    axes.set_ylabel('Longitude')
    axes.set_zlabel('Elevation')
    axes.set_zlim3d(bottom=0)

    # plt.xticks(fontsize=7, rotation = 45, **xticks)
    # plt.yticks(fontsize=7, **yticks)
    for t in axes.zaxis.get_major_ticks(): t.label.set_fontsize(7)

    plt.show()

    return fig