import numpy as np
import matplotlib.pyplot as plt


def tag(x, y, labels, **kwargs):
    """Add a label to a track at each point that it changes

    Parameters
    ----------
    x, y : array_like[float]
    labels : array_like[any]
        same length as x and y
    **kwargs
        Remaining keyword arguments are passed to `matplotlib.pyplot.text`
    """
    for label, idx in zip(np.unique(labels, return_index=True)):
        plt.text(x[idx], y[idx], str(label), **kwargs)
