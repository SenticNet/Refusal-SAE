import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_line(
    data,
    labels=None,
    xlabel="x",
    ylabel="y",
    title=None,
    x_tick=None,
    width=1200,
    height=600,
    hlines=None,  # New parameter for horizontal lines
    hline_labels=None,  # Labels for the horizontal lines
    save_path = None
):
    # Convert tensor→numpy
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    data = np.asarray(data)
    n_points = data.shape[-1]

    # Default x = integer indices
    if x_tick is None:
        x_tick = np.arange(n_points)

    fig = go.Figure()
    if data.ndim == 2:
        for i, row in enumerate(data):
            name = labels[i] if labels else f"Line {i}"
            fig.add_trace(go.Scatter(x=x_tick, y=row, mode="lines", name=name))
    else:
        fig.add_trace(go.Scatter(x=x_tick, y=data, mode="lines"))

    if hlines is not None:
        for idx, hline in enumerate(hlines):
            line_label = hline_labels[idx] if hline_labels else f"hline {idx}"
            fig.add_trace(go.Scatter(
                x=[x_tick[0], x_tick[-1]],
                y=[hline, hline],
                mode="lines",
                line=dict(dash="dash", width=2),
                name=line_label
            ))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=width,
        height=height
    )

    if x_tick is not None:
        if isinstance(x_tick[0], str):
            fig.update_xaxes(type="category", tickvals=x_tick)
        else:
            fig.update_xaxes(tickmode="array", tickvals=x_tick)

    fig.show()
    if save_path is not None:
        fig.write_image(save_path)

def plot_line_mpl(data,xlabel,ylabel,labels = None,x_tick = None,hlines = None, hline_labels = None,title=None,figsize=(10, 6),save_path = None):
    if isinstance(data,list):
        data = np.array(data)
    if data.ndim == 1:
        data = data.reshape(1,-1)
    plt.figure(figsize=figsize)
    if x_tick is None:
        x_tick = np.arange(data.shape[1])
    
    for i,d in enumerate(data):
        if labels is not None:
            plt.plot(x_tick,d,label = labels[i],linewidth=3)
    
    if hlines is not None:
        for i,h in enumerate(hlines):
            plt.plot([h] * len(x_tick),label = hline_labels[i],linestyle='--',linewidth=3)
    
    plt.xlabel(xlabel,fontsize=12)
    plt.ylabel(ylabel,fontsize=12)
    plt.xticks(x_tick,fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc = 'upper right',fontsize = 12)
    plt.grid(True)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_bar(data, x_tick=None, xlabel=None, ylabel=None, labels = None,title=None, figsize=(14, 6), tick_angle=45,line_plot = None,line_label = None,save_path = None):
    # Convert tensor→numpy if needed
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    data = np.asarray(data)
    
    # If no x_tick provided, use default indices (as strings)
    if x_tick is None:
        x_tick = [str(i) for i in range(len(data))]
    
    # Ensure x_tick and data have the same length
    if data.ndim == 1:
        x_range = range(len(data))
        if len(x_tick) != len(data):
            raise ValueError(f"Length mismatch: x_tick has {len(x_tick)} elements, data has {len(data)} elements")
            
            
    else:
        x_range = range(data.shape[1])
        if len(x_tick) != data.shape[1]:
            raise ValueError(f"Length mismatch: x_tick has {len(x_tick)} elements, data has {data.shape[1]} elements")
            
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the bar plot
    if data.ndim == 2:
        for i,row in enumerate(data):
            ax.bar(x_range, row, label=labels[i] if labels else f"Bar {i}",alpha = 0.7)
    else:
        ax.bar(x_range, data,label = labels)
    
    # Set the x-tick positions and labels
    ax.set_xticks(x_range)
    ax.set_xticklabels(x_tick, rotation=tick_angle, ha='right')
    ax.tick_params(axis='x', labelsize =14)
    ax.tick_params(axis='y', labelsize =14)

    if line_plot is not None : # overlap a line
        ax2 = ax.twinx()
        ax2.plot(x_range,line_plot,color = 'red',marker = 'o',label = line_label)
        ax2.tick_params(axis='y', labelcolor='red', labelsize = 14)
    
       
    # Add labels and title
    if xlabel is not None:
        ax.set_xlabel(xlabel,fontsize=12)
    if ylabel is not None:
        ax.set_ylabel(ylabel,fontsize=14)
    if title:
        ax.set_title(title,fontsize=14)
    
    # Adjust bottom margin to accommodate long tick labels
    plt.subplots_adjust(bottom=0.25)
    
    # Show the plot
    if labels is not None:
        if line_plot is not None:
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1.0))
            ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.93))
        else:
            ax.legend()
    plt.tight_layout()
    if not save_path:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()
    return None

def plot_bar_seaborn(data, x_tick=None, xlabel=None, ylabel=None, labels=None, title=None,
                     figsize=(14, 6), tick_angle=45, save_path=None,
                     line_plot=None, line_label=None, hlines=None, hline_labels=None,
                     show_grid=True, grid_linewidth=1.5):

    # Convert tensor to numpy
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    data = np.asarray(data)

    # Ensure 2D
    if data.ndim == 1:
        data = data[np.newaxis, :]

    num_bars = data.shape[1]
    if x_tick is None:
        x_tick = [str(i) for i in range(num_bars)]
    if len(x_tick) != num_bars:
        raise ValueError("x_tick must match number of columns in data")

    # Apply seaborn style
    sns.set_style('darkgrid' if show_grid else 'dark')
    plt.rc('grid', linewidth=grid_linewidth)

    # Prepare data
    if labels:
        df = pd.DataFrame(data.T, columns=labels)
        df['x'] = x_tick
        df_melt = df.melt(id_vars='x', var_name='Group', value_name='Value')
        use_hue = True
    else:
        df = pd.DataFrame({'x': x_tick, 'Value': data[0]})
        use_hue = False

    # Plotting
    plt.figure(figsize=figsize)
    if use_hue:
        ax = sns.barplot(data=df_melt, x='x', y='Value', hue='Group', estimator=sum, errorbar=None)
    else:
        ax = sns.barplot(data=df, x='x', y='Value', color=sns.color_palette()[0], errorbar=None)

    # X-axis formatting
    plt.xticks(rotation=tick_angle,fontsize=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14)
    else:
        ax.set_xlabel("")
        ax.xaxis.label.set_visible(False)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)

    if title:
        ax.set_title(title, fontsize=16)

    # Optional horizontal lines
    if hlines:
        for i, h in enumerate(hlines):
            label = hline_labels[i] if hline_labels and i < len(hline_labels) else None
            ax.axhline(h, linestyle='--', color='gray', linewidth=1.2, label=label)

    # Optional line plot
    ax2 = None
    if line_plot is not None:
        ax2 = ax.twinx()
        ax2.plot(x_tick, line_plot, color='red', marker='o', label=line_label)
        ax2.tick_params(axis='y', labelcolor='red', labelsize=12)
        # if line_label:
        #     ax2.legend(loc='upper right', fontsize=12)

    # Conditional legend on main axis
    if labels:
        if line_plot is not None:
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1.0))
            ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.94))
        else:
            ax.legend(title=None, fontsize=12,loc = 'upper right')
    elif hlines and hline_labels:
        ax.legend(fontsize=12)
    # else:
    #     ax.legend().remove()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_heatmap(data,x_tick,y_tick,xlabel,ylabel,title='None'):
    x_tick_vals = list(range(data.shape[1]))
    y_tick_vals = list(range(data.shape[0]))
    fig = go.Figure(data=go.Heatmap(
            z=data,
            x=x_tick_vals,  # assign tick values along the x-axis
            y=y_tick_vals,  # assign tick values along the y-axis
            colorscale='Viridis'  # choose a colorscale
        ))
    fig.update_layout(
    title=title,
    xaxis=dict(
        title=xlabel,
        tickmode='array',
        tickvals=x_tick_vals,
        ticktext=x_tick
    ),
    yaxis=dict(
        title=ylabel,
        tickmode='array',
        tickvals=y_tick_vals,
        ticktext=y_tick
        ),
    )

    fig.show()


def plot_tensor_scatter(data, x_tick, y_tick, xlabel,ylabel,labels=None, title='None',figsize = (8,6),markers = None,sharey=True): # if sharey, ytick is a single list else xtick is
    # If a single tensor is provided, wrap it in a list.
    if not isinstance(data, list):
        data = [data]
    # Default labels if not provided.
    if labels is None:
        labels = [f"Tensor {i+1}" for i in range(len(data))]
    
    fig, ax = plt.subplots(figsize=figsize)
    sc = None  # This will hold the last scatter for the colorbar.

    if not markers:
        markers = ['o','^','s','v','p']
        if len(data) > len(markers):
            raise ValueError("Not enough markers provided for the number of data, include your own markers")
    
    vmax = max([np.max(tensor) for tensor in data])
    vmin = min([np.min(tensor) for tensor in data])
    # Loop through each tensor and add it as a scatter trace.
    for sample_pos,(tensor, label) in enumerate(zip(data, labels)):
        ny, nx = tensor.shape
        sample_xtick = x_tick[sample_pos] if sharey else x_tick
        sameple_ytick = y_tick if sharey else y_tick[sample_pos]
        xs, ys, colors = [], [], []
        for i in range(ny):
            for j in range(nx):
                xs.append(sample_xtick[j])
                ys.append(i if isinstance(sameple_ytick[0], str) else sameple_ytick[i])
                colors.append(tensor[i, j])
        
        sc = ax.scatter(xs, ys, c=colors, cmap='viridis', label=label, s=75, edgecolors='k',marker = markers[sample_pos],vmax=vmax,vmin=vmin,alpha=0.7)
    
    # Set the x-axis limits to a free range (e.g., 0 to 100).
    # ax.set_xlim(0, 100)
    
    # Configure y-axis:
    # If y_tick are strings, use indices for positions and set the tick labels accordingly.
    if isinstance(y_tick[0], str):
        ax.set_yticks(range(len(y_tick)))
        ax.set_yticklabels(y_tick)
    else:
        ax.set_yticks(y_tick)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add a colorbar for the marker colors.
    plt.colorbar(sc, ax=ax,)
    
    # Place legend on top of the plot.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(data))
    
    plt.tight_layout()
    plt.show()