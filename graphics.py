# -*- coding: utf-8 -*-
# updated version of this file is maintained at
# https://github.com/shullgroup/QBKPy/blob/main/test/graphics.py

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def double_headed_arrow(ax, x1, y1, x2, y2, 
                        color='C0', linewidth=2, 
                        mutation_scale=15, 
                        zorder=3, 
                        **kwargs):
    """
    Draw a double-headed arrow from (x1, y1) to (x2, y2) on the given axes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    x1, y1, x2, y2 : float
        Coordinates in data space.
    color : str
        Arrow color.
    linewidth : float
        Line width.
    mutation_scale : float
        Controls the size of the arrowheads.
    zorder : int
        Drawing order.
    kwargs : dict
        Passed through to FancyArrowPatch (e.g., alpha, linestyle).
    """
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='<->',           # double-headed
        mutation_scale=mutation_scale,
        color=color,
        linewidth=linewidth,
        zorder=zorder,
        shrinkA=0.0, shrinkB=0.0,   # no shrinking at ends
        # By default, FancyArrowPatch uses data coordinates from the Axes
        **kwargs
    )
    ax.add_patch(arrow)
    return arrow

def vline(x, ax, **kwargs):
    '''
    Draws vertical line at x using existing limits

    Parameters
    ----------
    x : float
        location of the vertical line
    ax : axis
        axis on which to plot the line.

    Returns
    -------
    None.

    '''
    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]
    linestyle = kwargs.get('linestyle', 'solid')
    color = kwargs.get('color', 'k')
    label = kwargs.get('label', None)
    
    ax.vlines(x, ymin, ymax, color = color, linestyle = linestyle, label = label)

def hline(y, ax, **kwargs):
    '''
    Draws horizontal line at y using existing limits

    Parameters
    ----------
    y : float
        location of the horizontal line
    ax : axis
        axis on which to plot the line.

    Returns
    -------
    None.

    '''
    
    xmin = ax.get_xlim()[0]
    xmax = ax.get_xlim()[1]
    linestyle = kwargs.get('linestyle', '--')
    linewidth = kwargs.get('linewidth', 1.5)
    color = kwargs.get('color', 'k')

    ax.hlines(y, xmin, xmax, color = color, linestyle = linestyle,
              linewidth = linewidth)