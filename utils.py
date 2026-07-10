# -*- coding: utf-8 -*-
# updated version of this file is maintained at
# https://github.com/shullgroup/QBKPy/blob/main/utils.py

import numpy as np
import pandas as pd
import csv
from cycler import cycler
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib import rcParams
from pathlib import Path


# Shared across the library
cyclers = {'broderick':cycler(color=['#0093F5', '#F08E2C', '#000000', '#424EBD', 
                               '#B04D25', '#75CA85', '#C892D6']*3, 
                        linestyle=['-']*7 + ['--']*7 + [':']*7),
           'anisha':cycler(color=['purple', '#F28C28', '#D291BC'])}

def set_default_cycler(cycler=cyclers['broderick']):   
    rcParams['axes.prop_cycle'] = cycler

def is_numeric(cell):
    '''
    Check if a cell has numeric data. Used to start reading data files.
    
    Parameters
    ----------
    cell : str
        String which may be numeric (cell in a given row)
    
    Returns
    -------
    bool
        If cell is numeric (True) or not (False)
    '''
    try:
        float(cell)
        return True
    except (ValueError, TypeError):
        return False


from openpyxl import load_workbook


def first_line(path, **kwargs):
    """
    Determine the first row of numeric data for CSV or Excel files.
    Includes automatic delimiter detection and efficient Excel scanning.
    """

    path = Path(path)
    ext = path.suffix.lower()

    sep = kwargs.get('sep', None)  # None triggers auto-detection
    target_cols = kwargs.get('target_cols', None)
    encoding = kwargs.get('encoding', 'utf-8')

    # ------------------------------------------------------------
    # Excel files: use openpyxl to stream rows
    # ------------------------------------------------------------
    if ext in [".xls", ".xlsx"]:
        # For .xls you'd need xlrd or similar; this handles .xlsx well.
        if ext == ".xlsx":
            wb = load_workbook(path, read_only=True, data_only=True)
            ws = wb.active  # or choose a specific sheet

            # openpyxl rows are 1-based; we want 0-based index
            for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=0):
                cells = list(row)

                # Skip completely empty rows
                if all(c is None or str(c).strip() == "" for c in cells):
                    continue

                if target_cols is not None:
                    # Ensure we have enough columns
                    if len(cells) > max(target_cols):
                        if all(is_numeric(cells[c]) for c in target_cols):
                            return row_idx
                else:
                    if any(is_numeric(c) for c in cells):
                        return row_idx

            return 0
        else:
            # Simple fallback for .xls: load once with pandas
            df = pd.read_excel(path, header=None)
            for i, row in df.iterrows():
                cells = row.tolist()
                if not all(isinstance(x, (int, float)) for x in cells):
                    continue

                if target_cols is not None:
                    if all(is_numeric(cells[c]) for c in target_cols):
                        return int(i)
                else:
                    if any(is_numeric(c) for c in cells):
                        return int(i)
            return 0

    # ------------------------------------------------------------
    # Text-based files: CSV, TXT, DAT
    # ------------------------------------------------------------

    # Try UTF-8, fallback to latin-1
    try:
        with open(path, 'r', encoding=encoding) as f:
            f.readline()
    except UnicodeDecodeError:
        encoding = "latin-1"

    # Auto-detect delimiter if not provided
    if sep is None:
        with open(path, 'r', encoding=encoding) as f:
            sample = f.read(4096)
            try:
                sep = csv.Sniffer().sniff(sample).delimiter
            except Exception:
                sep = ','  # fallback

    with open(path, 'r', encoding=encoding) as f:
        reader = csv.reader(f, delimiter=sep)

        for i, row in enumerate(reader):
            cleaned = [cell.strip() for cell in row]

            if not cleaned:
                continue

            if target_cols is not None:
                if len(cleaned) > max(target_cols):
                    if all(is_numeric(cleaned[c]) for c in target_cols):
                        return i
            else:
                if any(is_numeric(cell) for cell in cleaned):
                    return i

    return 0




def remove_step_lines(df):
    '''
    Remove extra lines in files from multiple steps in 
    TA Instruments experiments
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe from experimental file
    
    Returns
    -------
    df : pd.DataFrame
        Cleaned dataframe with step lines removed.
    '''

    to_drop = []
    for rows in np.arange(0,len(df)):
        
        if df.iloc[rows, 0] == '[step]':
            to_drop.extend([rows,rows+1,rows+2,rows+3])
    
    df = df.drop(to_drop).reset_index(drop=True)

    return df



def read_data_file(path, **kwargs):
    """
    Read a general data file (.csv, .xls, .xlsx) by detecting file type.
    For text files, find the first line of numeric data and return a DataFrame.
    Ensures all values are converted to numeric when possible.
    """

    path = Path(path)
    ext = path.suffix.lower()

    sep = kwargs.get('sep', '\t')
    sheet_name = kwargs.get('sheet_name', None)
    target_cols = kwargs.get('target_cols', [0, 1])
    names = kwargs.get('names', ['col1', 'col2'])

    # Determine where numeric data begins
    skiprows = first_line(path, sep=sep, target_cols=target_cols)

    # ------------------------------------------------------------
    # Excel files
    # ------------------------------------------------------------
    if ext in [".xls", ".xlsx"]:
        df = pd.read_excel(
            path,
            usecols=target_cols,
            names=names,
            skiprows=skiprows,
            header=None,
            sheet_name = sheet_name
        )

    # ------------------------------------------------------------
    # CSV / text files
    # ------------------------------------------------------------
    elif ext in [".csv", ".txt", ".dat"]:
        df = pd.read_csv(
            path,
            delimiter=sep,
            skiprows=skiprows,
            usecols=target_cols,
            names=names
        )

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # ------------------------------------------------------------
    # Drop empty rows
    # ------------------------------------------------------------
    df = df.dropna(how="all").reset_index(drop=True)

    # ------------------------------------------------------------
    # Convert all columns to numeric where possible
    # ------------------------------------------------------------
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df



def downsample_points_per_decade(
    x,
    y=None,
    points_per_decade=10,
    base=10.0,
    include_endpoints=True,
):
    """
    Downsample data to approximately `points_per_decade` per logarithmic
    decade in x.

    Parameters
    ----------
    x : array-like, shape (n,)
        Independent variable. Must be strictly positive (for log scaling).
    y : None | array-like, shape (n,) or (n, m), optional
        Dependent variable(s) aligned with x. If provided with shape (n, m),
        each column is treated as a separate series sharing the same x.
    points_per_decade : int, default=10
        Target number of retained points per decade (in log-base `base`).
        If a decade contains fewer than this many original points, they are
        all kept.
    base : float, default=10.0
        Logarithm base that defines a “decade”. Use 10 for log10 decades,
        or e.g. `np.e` for natural-log-based “decades”.
    include_endpoints : bool, default=True
        If True, always include the global first and last indices, and try
        to include integer decade boundaries when present.

    Returns
    -------
    x_ds : ndarray, shape (k,)
        Downsampled x (sorted ascending).
    y_ds : None | ndarray
        Downsampled y aligned to x_ds. Returns None if y is None.
        If input y was (n, m), output is (k, m).
    idx : ndarray, shape (k,)
        Indices into the original arrays for the retained samples.

    Notes
    -----
    - This keeps actual data points; it does NOT interpolate.
    - Selection is done by choosing points closest to evenly spaced targets
      in log-space within each integer decade [d, d+1), except the last
      decade which includes its upper bound.
    - Duplicates are removed globally and order is preserved.
    """
    x = np.asarray(x)
    if np.any(x <= 0):
        raise ValueError("All x values must be > 0 for logarithmic decades.")
    if x.ndim != 1:
        raise ValueError("x must be 1-D.")

    n = x.size
    if y is not None:
        y = np.asarray(y)
        if y.shape[0] != n:
            raise ValueError(
                "x and y must have the same length along axis 0."
            )

    # Work in log-base `base`
    logx = np.log(x) / np.log(base)

    # Identify integer-decade bins covering the data
    d_min = int(np.floor(np.min(logx)))
    d_max = int(np.floor(np.max(logx)))  # inclusive starting decade

    selected_idx = []

    for d in range(d_min, d_max + 1):
        # For all but the last decade, use [d, d+1); for the last, include
        # right edge
        if d < d_max:
            mask = (logx >= d) & (logx < d + 1)
            right_endpoint_included = False
        else:
            mask = (logx >= d) & (logx <= d + 1)
            right_endpoint_included = True

        idx_in_decade = np.nonzero(mask)[0]
        if idx_in_decade.size == 0:
            continue

        # If there are already fewer points than target, keep them all
        if idx_in_decade.size <= points_per_decade:
            selected_idx.extend(idx_in_decade.tolist())
            continue

        # Target evenly spaced positions in log space for this decade
        if right_endpoint_included:
            targets = np.linspace(
                d, d + 1, points_per_decade, endpoint=False
            )
        else:
            targets = np.linspace(
                d, d + 1, points_per_decade, endpoint=False
            )

        # For each target, pick the index whose logx is closest
        logx_dec = logx[idx_in_decade]
        for t in targets:
            j = np.argmin(np.abs(logx_dec - t))
            selected_idx.append(idx_in_decade[j])

        # Optionally include decade boundaries if they exist in the data
        if include_endpoints:
            # Left boundary at d
            j_left = np.where(
                np.isclose(logx_dec, d, rtol=0, atol=1e-12)
            )[0]
            if j_left.size > 0:
                selected_idx.append(idx_in_decade[j_left[0]])
            # Right boundary at d+1 (only meaningful when last decade or
            # if a data point hits exactly)
            if right_endpoint_included:
                j_right = np.where(
                    np.isclose(logx_dec, d + 1, rtol=0, atol=1e-12)
                )[0]
                if j_right.size > 0:
                    selected_idx.append(idx_in_decade[j_right[0]])

    # Always include global endpoints if requested
    if include_endpoints:
        selected_idx.extend([0, n - 1])

    # Deduplicate while preserving order
    seen = set()
    ordered_unique_idx = []
    for i in selected_idx:
        if i not in seen:
            seen.add(i)
            ordered_unique_idx.append(i)
    ordered_unique_idx = np.array(ordered_unique_idx, dtype=int)

    # Sort indices by original index to preserve input sequence
    ordered_unique_idx.sort()

    x_ds = x[ordered_unique_idx]
    if y is None:
        y_ds = None
    else:
        y_ds = y[ordered_unique_idx, ...]

    return x_ds, y_ds, ordered_unique_idx


def downsample_df_per_decade(
    df: pd.DataFrame,
    column: str,
    points_per_decade: int = 10,
    base: float = 10.0,
    include_endpoints: bool = False,
):
    """
    Downsample DataFrame rows to approximately `points_per_decade`
    per logarithmic decade of the specified column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    column : str
        Name of the column used for logarithmic downsampling.
        Values must be strictly positive.
    points_per_decade : int, default=10
        Target number of retained points per decade.
    base : float, default=10.0
        Logarithm base that defines a “decade”.
    include_endpoints : bool, default=False
        If True, always retain the first and last rows and try to
        include integer decade boundaries when present.

    Returns
    -------
    df_ds : pandas.DataFrame
        Downsampled DataFrame (rows preserved, no interpolation).
    idx : ndarray
        Integer indices of retained rows (relative to original df).

    Notes
    -----
    - Keeps actual data rows; does NOT interpolate.
    - Selection is made in log-space within each integer decade.
    - Row order is preserved.
    """

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    x = df[column].to_numpy()

    if np.any(x <= 0):
        raise ValueError(
            f"All values in column '{column}' must be > 0 "
            "for logarithmic decades."
        )

    # Work in log-base `base`
    logx = np.log(x) / np.log(base)

    d_min = int(np.floor(logx.min()))
    d_max = int(np.floor(logx.max()))

    selected_idx = []

    for d in range(d_min, d_max + 1):
        if d < d_max:
            mask = (logx >= d) & (logx < d + 1)
            right_inclusive = False
        else:
            mask = (logx >= d) & (logx <= d + 1)
            right_inclusive = True

        idx_in_decade = np.nonzero(mask)[0]
        if idx_in_decade.size == 0:
            continue

        # Keep all points if already sparse
        if idx_in_decade.size <= points_per_decade:
            selected_idx.extend(idx_in_decade.tolist())
            continue

        # Evenly spaced targets in log-space
        targets = np.linspace(
            d, d + 1, points_per_decade, endpoint=False
        )

        logx_dec = logx[idx_in_decade]

        for t in targets:
            j = np.argmin(np.abs(logx_dec - t))
            selected_idx.append(idx_in_decade[j])

        if include_endpoints:
            # Left decade boundary
            j_left = np.where(np.isclose(logx_dec, d, atol=1e-12))[0]
            if j_left.size:
                selected_idx.append(idx_in_decade[j_left[0]])

            # Right decade boundary (only for last decade)
            if right_inclusive:
                j_right = np.where(
                    np.isclose(logx_dec, d + 1, atol=1e-12)
                )[0]
                if j_right.size:
                    selected_idx.append(idx_in_decade[j_right[0]])

    # Always include global endpoints
    if include_endpoints and len(df):
        selected_idx.extend([0, len(df) - 1])

    # Deduplicate while preserving order
    seen = set()
    ordered_idx = []
    for i in selected_idx:
        if i not in seen:
            seen.add(i)
            ordered_idx.append(i)

    ordered_idx = np.array(ordered_idx, dtype=int)
    ordered_idx.sort()

    return df.iloc[ordered_idx].copy()


def baseline_correct(df, x_col, y_col, x_range=None):
    """
    Apply linear baseline correction to q vs temp using two anchor temperatures.

    Parameters
    ----------
    df : pandas.DataFrame
    y : str
        Name of y data column
    x_col : str
        Name of a data column
    x_range : list or tuple (t1, t2)
        Two x values used to determine baseline

    Returns
    -------
    df_out : pandas.DataFrame
        Copy of dataframe with added 'y_corrected' column
    """

    if x_range is None or len(x_range) != 2:
        raise ValueError("x_range must be a two-element list or tuple")

    x1, x2 = x_range

    # Select rows at (or nearest to) the two temperatures
    idx1 = (df[x_col] - x1).abs().idxmin()
    idx2 = (df[x_col] - x2).abs().idxmin()

    # Extract anchor points
    x = df.loc[[idx1, idx2], x_col].values
    y = df.loc[[idx1, idx2], y_col].values

    # Fit line (slope m, intercept b)
    m, b = np.polyfit(x, y, 1)

    # Compute baseline for all temps
    baseline = m * df[x_col] + b

    # Subtract baseline
    df_out = df.copy()
    df_out[y_col] = df[y_col] - baseline

    return df_out


def add_scaled_right_axis(ax, factor, label, precision=1):
    """
    Create a right-side twin axis whose scale is linearly related
    to the left axis by a user-specified factor.

    Right axis value = Left axis value * factor
    """
    ax_right = ax.twinx()

    # Format ticks on the right axis
    ax_right.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"{y:.{precision}f}")
    )
    ax_right.set_ylabel(label)

    # --- Synchronize limits dynamically ---
    def sync_right_axis(ax):
        left_min, left_max = ax.get_ylim()
        ax_right.set_ylim(left_min * factor, left_max * factor)

    # Initial sync
    sync_right_axis(ax)

    # Sync whenever the left axis changes (zoom/pan)
    ax.callbacks.connect("ylim_changed", sync_right_axis)

    return ax_right

