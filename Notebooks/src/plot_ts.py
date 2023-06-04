import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_ts(true, predicted, ts_start, ts_end):
    """This function plots the prediction of a timeseries versus the real series.

    Args:
        true (_type_): The true values
        predicted (_type_): The predicted values
        ts_start (int): The index of the first predicted value to print
        ts_end (int): The index of the last predicted value to print
    """

    ts_len = ts_end - ts_start

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=list(range(ts_start, ts_end)),
            y=predicted[:ts_len],
            name="predictions",
            opacity=1,
        )
    )

    fig.add_trace(
        go.Bar(
            x=list(range(ts_start, ts_end)),
            y=true[:ts_len],
            name="true values",
            opacity=1,
        )
    )

    fig.show()
    pass


def plot_loss(history):
    """This function plots the loss of a Tensorflow model.

    Args:
        history (_type_): The history from model.fit
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]], subplot_titles=['MSE loss'], x_title='Epochs')
    fig.add_trace(
        go.Scatter(
            x=list(range(len(history.history["loss"]))),
            y=history.history["loss"],
            name="Training loss",
            opacity=1,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(len(history.history["val_loss"]))),
            y=history.history["val_loss"],
            name="Validation loss",
            opacity=1,
        )
    )

    fig.show()

def plot_correlation(history):
    """This function plots the correlation coefficient of a Tensorflow model.

    Args:
        history (_type_): The history from model.fit
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]], subplot_titles=['Correlation'], x_title='Epochs')
    fig.add_trace(
        go.Scatter(
            x=list(range(len(history.history['correlation_coefficient']))),
            y=history.history['correlation_coefficient'],
            name="Training Correlation",
            opacity=1,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(len(history.history['val_correlation_coefficient']))),
            y=history.history['val_correlation_coefficient'],
            name="Validation Correlation",
            opacity=1,
        )
    )
    fig.show()
