

# %%
import re
import csv
from pathlib import Path
import codecs
import tomllib
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)

# logging.basicConfig(level=logging.INFO) 


@dataclass
class ColumnMapping:
    raw_name: str             # string of header in header row
    raw_unit: str             # string of unit in unit row
    standard_name: str          # Unified database/internal standard name
    short_name: str          # Unified short name or abbreviation
    si_unit: str              # Target SI unit label
    # A function to apply the SI transformation. Defaults to identity (no change).
    convert_fn: Callable[[pd.Series], pd.Series] = lambda x: x

def current_file_directory() -> Path:
    """
    Returns the directory of the current script file.
    """
    return Path(__file__).resolve().parent


def registry_file_path(registry_name):
    '''
    return full file path by registry_name

    '''
    # Get the directory containing the current script file
    current_directory = current_file_directory()

    # Example: Constructing a path to your TOML file in the same directory
    registry_path = current_directory/ 'registries' / f'{registry_name}.toml'
    logger.info(f'registry_path: {registry_path}')
    
    if registry_path.is_file():
        logger.info("The file exists!")
        return registry_path
    else:
        logger.warning("File not found or the path is a directory.")
        return None


def load_registry_from_toml(registry_name: str) -> dict | None:
    '''
    Loads column configuration settings from a TOML file and builds the registry.
    '''
    registry_path = registry_file_path(registry_name)
    if not registry_path:
        return None
    
    with open(registry_path, "rb") as f:
        config_data = tomllib.load(f)
        
    registry = {}
    for column_title, metadata in config_data.items():
        raw_name = metadata["raw_name"]
        raw_unit = metadata["raw_unit"]
        standard_name = metadata["standard_name"]
        short_name = metadata["short_name"]
        si_unit = metadata["si_unit"]
        
        # Check for our conversion math formula strings
        if "convert_expr" in metadata and metadata["convert_expr"]:
            expr_str = metadata["convert_expr"]
            # Safely inject numpy into the evaluation environment for np.pi math
            convert_fn = eval(f"lambda x: {expr_str}", {"np": np})
        else:
            convert_fn = lambda x: x
            
        registry[column_title] = ColumnMapping(
            raw_name=raw_name,
            raw_unit=raw_unit,
            standard_name=standard_name,
            short_name=short_name,
            si_unit=si_unit,
            convert_fn=convert_fn
        )
    return registry


def get_file_encoding(file_path: str | Path = 'utf-8') -> str:
    '''
    Checks a text file for standard Byte Order Marks (BOM) to precisely 
    identify UTF encodings. Falls back to detecting cp1252/utf-8 if no BOM exists.
    '''
    path = Path(file_path)
    
    # Read the first 4 bytes to check for a Byte Order Mark (BOM)
    with open(path, 'rb') as f:
        raw_bytes = f.read(4)
        
    # Check against standard BOM signatures
    if raw_bytes.startswith(codecs.BOM_UTF32_BE):
        return 'utf-32-be'
    if raw_bytes.startswith(codecs.BOM_UTF32_LE):
        return 'utf-32-le'
    if raw_bytes.startswith(codecs.BOM_UTF16_BE):
        return 'utf-16-be'
    if raw_bytes.startswith(codecs.BOM_UTF16_LE):
        return 'utf-16-le'
    if raw_bytes.startswith(codecs.BOM_UTF8):
        return 'utf-8-sig' # 'utf-8-sig' automatically strips the BOM character when reading
        
    # If no BOM is present, check if it can be successfully decoded as UTF-8
    try:
        with open(path, 'r', encoding='utf-8') as f:
            f.read(1024) # Test read a chunk
        return 'utf-8'
    except UnicodeDecodeError:
        # If UTF-8 fails, it's highly likely a legacy Windows export format
        return 'cp1252'


def get_text_delimiter(lines: list[str]) -> str:
    """
    Determines the delimiter of a text or CSV file from a list of its lines.
    Handles standard punctuation (',', ';', '\t') and whitespace variants ('\\s', '\\s+').
    
    Parameters:
        lines (list of str): The raw text lines read from the data file.
        
    Returns:
        str: The detected delimiter string compatible with pandas.read_csv()
    """
    if not lines:
        return '\t'  # Safe default fallback for empty files

    # 1. Clean up and extract candidate lines that actually contain data/headers
    # Filter out pure blank rows or whitespace padding
    sample_lines = [line.strip() for line in lines if line.strip()]
    if not sample_lines:
        return '\t'
        
    # We use the last few valid rows as a combined context block for parsing
    sample_block = "\n".join(sample_lines[min(len(lines), 10) * -1:])

    # 2. Strategy A: Try the standard CSV Sniffer for common structural punctuation
    try:
        # Explicitly look for standard lab delimiters: tab, comma, semicolon
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_block, delimiters=[',', ';', '\t'])
        return dialect.delimiter
    except csv.Error:
        # If sniffing fails, it's highly likely a non-standard whitespace-delimited file
        pass

    # 3. Strategy B: Analyze whitespace structures using RegEx matching frequency
    # We inspect the target header line (often the first line with real text)
    target_line = sample_lines[0]

    # Count how many strict single tabs or semicolons might have slipped through
    if target_line.count('\t') > 0:
        return '\t'
    if target_line.count(';') > 0:
        return ';'

    # Check for uniform multi-space columns vs single spaces
    multi_space_matches = re.findall(r'\s{2,}', target_line)
    single_space_matches = re.findall(r' ', target_line)

    if len(multi_space_matches) > 0:
        # Consecutive variable space blocks separating tokens (e.g., space-aligned tables)
        return r'\s+'
    elif len(single_space_matches) > 0:
        # Single whitespace characters mapping explicit columns
        return r'\s'

    # Fallback default if the file contains no distinct separating punctuation/spacing
    return '\t'


def get_file_config(file_path, registry_name, encoding=None, delim=None, errors='ignore'):
    """
    Determines pandas.read_csv parameters sequentially:
    1. Finds total data columns from the bottom 2 non-empty lines.
    2. Identifies header index via registry 'raw_name' overlap.
    3. Identifies data start via first row containing a numeric primary cell.
    4. Confirms unit line inside the gap boundary using registry 'raw_unit' lookups.
    """
    path = Path(file_path)
    
    registry = load_registry_from_toml(registry_name)

    # Gather file format metrics and lines
    if not encoding:
        encoding = get_file_encoding(path)
    with open(path, 'r', encoding=encoding, errors=errors) as f:
        lines = f.readlines()
    
    if not delim:
        delim = get_text_delimiter(lines)

    def split_to_tokens(line_str):
        """Splits an input string cleanly by the file-level delimiter."""
        return [t.strip().strip('\'"').strip() for t in line_str.strip('\n').split(delim)]

    # -------------------------------------------------------------------------
    # STAGE 1: Find the number of columns using the last 2 non-empty rows
    # -------------------------------------------------------------------------
    valid_bottom_rows = []
    for i in range(len(lines) - 1, -1, -1):
        tokens = split_to_tokens(lines[i])
        if any(tokens):
            valid_bottom_rows.append(tokens)
            if len(valid_bottom_rows) == 2:
                break

    if len(valid_bottom_rows) < 2:
        raise ValueError(f"File '{path.name}' does not contain enough data records.")
    
    if len(valid_bottom_rows[0]) != len(valid_bottom_rows[1]):
        raise ValueError(f"Column width mismatch at the bottom of '{path.name}'.")
        
    num_columns = len(valid_bottom_rows[0])
    logger.info(f"Detected {num_columns} columns in '{path.name}'")
    logger.info(f'last row: {valid_bottom_rows[0]}')
    logger.info(f'lines[{i}]: {lines[i]}')

    # all names and units
    registered_raw_names = set()
    registered_raw_units = set()
    for item in registry.values():
        name = item.get('raw_name') if isinstance(item, dict) else getattr(item, 'raw_name', None)
        unit = item.get('raw_unit') if isinstance(item, dict) else getattr(item, 'raw_unit', None)
        if name: registered_raw_names.add(name)
        if unit: registered_raw_units.add(unit)

    header_line_idx = None
    first_numeric_idx = None
    units_line_idx = None

    # -------------------------------------------------------------------------
    # STAGE 2: Find the header row via registry name verification
    # -------------------------------------------------------------------------
    for i, line in enumerate(lines):
        tokens = split_to_tokens(line)
        if len(tokens) != num_columns:
            continue
            
        # sliced_tokens = tokens[-num_columns:]
        # name_matches = sum(1 for t in sliced_tokens if t in registered_raw_names)
        name_matches = sum(1 for t in tokens if t in registered_raw_names)
        logger.info(f'{i} tokens: {tokens}')
        logger.info(f'{i} name_matches: {name_matches}')
        if name_matches >= 2:
            header_line_idx = i
            break

    if header_line_idx is None:
        raise ValueError(f"Could not validate header elements using registry raw_names in '{path.name}'.")

    # -------------------------------------------------------------------------
    # STAGE 3: Find the first numeric data row (bypassing leading indents)
    # -------------------------------------------------------------------------
    numeric_pattern = re.compile(r'^\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*$')
    
    for i in range(header_line_idx + 1, len(lines)):
        tokens = split_to_tokens(lines[i])
        real_tokens = [t for t in tokens if t] # Filter out blank indent tokens
        
        if real_tokens and numeric_pattern.match(real_tokens[0]):
            first_numeric_idx = i
            break

    if first_numeric_idx is None:
        raise ValueError(f"Could not identify the start of the numeric data matrix in '{path.name}'.")

    # -------------------------------------------------------------------------
    # STAGE 4: Loop between them to find unit row using strict registry confirmation
    # -------------------------------------------------------------------------
    if first_numeric_idx > (header_line_idx + 1):
        for i in range(header_line_idx + 1, first_numeric_idx):
            tokens = split_to_tokens(lines[i])
            
            # Align right-sided layout matching data shape
            sliced_tokens = tokens[-num_columns:]
            
            # Count explicit presence of raw registry strings inside the slice
            unit_matches = sum(1 for t in sliced_tokens if t in registered_raw_units)
            
            # If it has the right layout dimensions and matches known strings, confirm it
            if unit_matches >= 1:
                units_line_idx = i
                break

    # -------------------------------------------------------------------------
    # STAGE 5: Reconstruct columns safely for Pandas
    # -------------------------------------------------------------------------
    headers = split_to_tokens(lines[header_line_idx])[-num_columns:]
    units = split_to_tokens(lines[units_line_idx])[-num_columns:] if units_line_idx is not None else [''] * num_columns

    clean_columns = []
    for h, u in zip(headers, units):
        if u:
            # Reassembles matching keys perfectly (e.g. "Angular Frequency [rad/s]")
            clean_columns.append(f"{h} {u}")
        else:
            clean_columns.append(h)

    return {
        'delimiter': delim,
        'skiprows': first_numeric_idx,
        'names': clean_columns,
        'encoding': encoding
    }


def read_csv_with_config(file_path, csv_config):

    # 6. Load the data using the extracted parameters
    df = pd.read_csv(
        file_path,
        sep=csv_config['delimiter'],
        skiprows=csv_config['skiprows'],
        names=csv_config['names'],
        header=None,
        encoding=csv_config['encoding'],
    )

    return df


def transform_df_to_si(df: pd.DataFrame, registry_name: str) -> pd.DataFrame:
    """
    Transforms DataFrame columns and numeric values into standardized SI units 
    using the registry map definition.
    """
    si_df = df.copy()
    rename_map = {}

    # get registry
    registry = load_registry_from_toml(registry_name)
    
    for col in df.columns:
        if col in registry:
            mapping = registry[col]
            
            # Apply unit conversion function if data is numeric
            if pd.api.types.is_numeric_dtype(si_df[col]):
                si_df[col] = mapping.convert_fn(si_df[col])
            
            # Standardize names as 'common_name_unit' (e.g., 'torque_N·m', 'shear_strain_1')
            if mapping.si_unit:
                rename_map[col] = f"{mapping.standard_name}_{mapping.si_unit}"
            else:
                rename_map[col] = f"{mapping.standard_name}"
        else:
            # Fallback for unexpected columns
            rename_map[col] = col.lower().replace(" ", "_")
            
    return si_df.rename(columns=rename_map)


# function in one
def read_csv_to_si(file_path: str | Path, registry_name: str, encoding=None, delim=None):
    '''
    
    '''
    csv_config = get_file_config(file_path, registry_name, encoding=encoding, delim=delim)
    df = read_csv_with_config(file_path, csv_config)
    df = transform_df_to_si(df,registry_name)

    return df

# %%
