import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(
    matrix,
    classes,
    figure_name,
    ax=None,
    cbar_kw={},
    cbarlabel="",
    normalize=True,
    **kwargs
):
    """
    Create a heatmap from a numpy array and a list of class labels.

    Parameters
    ----------
    matrix
        A 2D numpy array of shape (N, M).
    classes
        A list or array of length N with the labels for the row and columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    normalize
        Whether normalize the confusion matrix to proportion. Default = True.
    **kwargs
        All other arguments are forwarded to `imshow`.

    Reference
    ---------
    https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """
    if not ax:
        ax = plt.gca()

    cmap = plt.cm.coolwarm
    if normalize:
        matrix = matrix.astype('float') / (matrix.sum(axis=1) + 1e-7)[:, np.newaxis]

    im = ax.imshow(matrix, interpolation='nearest', cmap=cmap, **kwargs)

    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(figure_name)
