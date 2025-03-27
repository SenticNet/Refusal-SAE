import plotly.graph_objects as go
import numpy as np
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
    hline_labels=None  # Labels for the horizontal lines
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
            fig.update_xaxes(tickmode="category", tickvals=x_tick)
        else:
            fig.update_xaxes(tickmode="array", tickvals=x_tick)

    fig.show()


def plot_bar(
    data,
    x_tick=None,
    xlabel="x",
    ylabel="y",
    title=None,
    width=1200,
    height=600,
    tick_angle=45
):
    # Convert tensor→numpy if needed
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    data = np.asarray(data)
    
    # If no x_tick provided, use default indices (as strings)
    if x_tick is None:
        x_tick = [str(i) for i in range(len(data))]
    
    # Create bar plot using string x values
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_tick, y=data))
    
    # Update layout: set labels, title, dimensions, and rotate xticks
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=width,
        height=height
    )
    fig.update_xaxes(tickangle=tick_angle)
    
    fig.show()