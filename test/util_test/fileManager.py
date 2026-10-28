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


@dataclass
class ColumnMapping:
    raw_name: str             # Original header string in raw data file
    raw_unit: str             # Original unit string in raw data file
    standard_name: str        # Unified internal variable name
    short_name: str           # Abbreviated key or symbol
    si_unit: str              # Standardized target SI unit label
    symbol_str: str = ""      # LaTeX string representation for plotting symbol (e.g., "$\\omega$")
    unit_str: str = ""        # LaTeX string representation for plotting unit (e.g., "$\\text{rad/s}$")
    # Function to apply the SI transformation. Defaults to identity (no-op).
    convert_fn: Callable[[pd.Series], pd.Series] = lambda x: x


def current_file_directory() -> Path:
    """Returns the directory of the current script file."""
    return Path(__file__).resolve().parent


def file_exists(file_path: str | Path) -> bool:
    """Checks if file exists."""
    return Path(file_path).is_file()


def registry_file_path(registry_name: str) -> Path | None:
    """Returns full file path by registry_name."""
    current_directory = current_file_directory()
    registry_path = current_directory / 'registries' / f'{registry_name}.toml'
    logger.info(f'registry_path: {registry_path}')
    
    if registry_path.is_file():
        logger.info("Registry file found!")
        return registry_path
    else:
        logger.warning(f"Registry file '{registry_name}.toml' not found.")
        return None


def load_standards_registry() -> dict:
    """Loads standards.toml containing unified symbols and LaTeX plotting units."""
    standards_path = current_file_directory() / 'registries' / 'standards.toml'
    if not standards_path.is_file():
        logger.warning("standards.toml not found. LaTeX plotting strings will default to empty.")
        return {}
    
    with open(standards_path, "rb") as f:
        return tomllib.load(f)


def load_registry_from_toml(registry_name: str) -> tuple[dict, dict[str, ColumnMapping]]:
    """
    Loads column configuration settings and file hints from a instrument TOML file.
    Merges symbol_str and unit_str from standards.toml based on [file].test_method.
    
    :return: Tuple of (file_hints_dict, registry_mapping_dict)
    """
    registry_path = registry_file_path(registry_name)
    if not registry_path:
        raise ValueError(f"Registry '{registry_name}' is missing or invalid.")
    
    with open(registry_path, "rb") as f:
        config_data = tomllib.load(f)

    # 1. Extract [file] header configuration block
    file_hints = config_data.get("file", {})
    test_method = file_hints.get("test_method", "").lower()

    # 2. Load global standard symbols & plot strings from standards.toml
    standards_db = load_standards_registry()
    method_standards = standards_db.get(test_method, {})

    registry = {}
    
    # 3. Parse [[vars]] section
    vars_list = config_data.get("vars", [])
    for metadata in vars_list:
        raw_name = metadata.get("raw_name", "")
        raw_unit = metadata.get("raw_unit", "")
        standard_name = metadata.get("standard_name", "")
        short_name = metadata.get("short_name", "")
        si_unit = metadata.get("si_unit", "")

        # Lookup plot strings from standards.toml matching standard_name
        std_meta = method_standards.get(standard_name, {})
        symbol_str = std_meta.get("symbol_str", "")
        unit_str = std_meta.get("unit_str", "")

        # Build math conversion function
        if "convert_expr" in metadata and metadata["convert_expr"]:
            expr_str = metadata["convert_expr"]
            convert_fn = eval(f"lambda x: {expr_str}", {"np": np})
        else:
            convert_fn = lambda x: x

        mapping = ColumnMapping(
            raw_name=raw_name,
            raw_unit=raw_unit,
            standard_name=standard_name,
            short_name=short_name,
            si_unit=si_unit,
            symbol_str=symbol_str,
            unit_str=unit_str,
            convert_fn=convert_fn
        )

        # Construct key matching full combined column header string (e.g. "Time [min]")
        lookup_key = f"{raw_name} {raw_unit}".strip() if raw_unit else raw_name
        registry[lookup_key] = mapping

    return file_hints, registry


def get_file_encoding(file_path: str | Path = 'utf-8') -> str:
    """Identifies UTF / legacy encoding from BOM or chunk decoding."""
    path = Path(file_path)
    
    with open(path, 'rb') as f:
        raw_bytes = f.read(4)
        
    if raw_bytes.startswith(codecs.BOM_UTF32_BE): return 'utf-32-be'
    if raw_bytes.startswith(codecs.BOM_UTF32_LE): return 'utf-32-le'
    if raw_bytes.startswith(codecs.BOM_UTF16_BE): return 'utf-16-be'
    if raw_bytes.startswith(codecs.BOM_UTF16_LE): return 'utf-16-le'
    if raw_bytes.startswith(codecs.BOM_UTF8):     return 'utf-8-sig'
        
    try:
        with open(path, 'r', encoding='utf-8') as f:
            f.read(1024)
        return 'utf-8'
    except UnicodeDecodeError:
        return 'cp1252'


def get_text_delimiter(lines: list[str]) -> str:
    """Determines delimiter of text file from line context."""
    if not lines:
        return '\t'

    sample_lines = [line.strip() for line in lines if line.strip()]
    if not sample_lines:
        return '\t'
        
    sample_block = "\n".join(sample_lines[min(len(lines), 10) * -1:])

    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_block, delimiters=',;\t')
        return dialect.delimiter
    except csv.Error:
        pass

    target_line = sample_lines[0]
    if target_line.count('\t') > 0: return '\t'
    if target_line.count(';') > 0:  return ';'

    if len(re.findall(r'\s{2,}', target_line)) > 0: return r'\s+'
    if len(re.findall(r' ', target_line)) > 0:      return r'\s'

    return '\t'


def get_file_config(file_path: str | Path, registry_name: str, encoding=None, delim=None, errors='ignore') -> dict:
    """
    Determines pandas read parameters by detecting matrix bounds and matching 
    headers/units against registry definitions.
    """
    path = Path(file_path)
    file_hints, registry = load_registry_from_toml(registry_name)

    # Use file_hints defaults if explicit overrides aren't provided
    if not encoding:
        encoding = file_hints.get("encoding") or get_file_encoding(path)
    if not delim:
        delim = file_hints.get("delim") or None

    with open(path, 'r', encoding=encoding, errors=errors) as f:
        lines = f.readlines()
    
    if not delim:
        delim = get_text_delimiter(lines)

    def split_to_tokens(line_str):
        return [t.strip().strip('\'"').strip() for t in line_str.strip('\n').split(delim)]

    # 1. Find column count from last non-empty rows
    valid_bottom_rows = []
    for i in range(len(lines) - 1, -1, -1):
        tokens = split_to_tokens(lines[i])
        if any(tokens):
            valid_bottom_rows.append(tokens)
            if len(valid_bottom_rows) == 2:
                break

    if len(valid_bottom_rows) < 2 or len(valid_bottom_rows[0]) != len(valid_bottom_rows[1]):
        raise ValueError(f"Invalid record structure in '{path.name}'.")
        
    num_columns = len(valid_bottom_rows[0])
    
    registered_raw_names = {item.raw_name for item in registry.values()}
    registered_raw_units = {item.raw_unit for item in registry.values() if item.raw_unit}

    header_line_idx = None
    first_numeric_idx = None
    units_line_idx = None

    # 2. Find header row
    for i, line in enumerate(lines):
        tokens = split_to_tokens(line)
        if len(tokens) != num_columns:
            continue
        if sum(1 for t in tokens if t in registered_raw_names) >= 2:
            header_line_idx = i
            break

    if header_line_idx is None:
        raise ValueError(f"Could not validate header elements using registry in '{path.name}'.")

    # 3. Find first numeric row
    numeric_pattern = re.compile(r'^\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*$')
    for i in range(header_line_idx + 1, len(lines)):
        tokens = split_to_tokens(lines[i])
        real_tokens = [t for t in tokens if t]
        if real_tokens and numeric_pattern.match(real_tokens[0]):
            first_numeric_idx = i
            break

    if first_numeric_idx is None:
        raise ValueError(f"Could not locate start of numeric data in '{path.name}'.")

    # 4. Locate unit row
    if first_numeric_idx > (header_line_idx + 1):
        for i in range(header_line_idx + 1, first_numeric_idx):
            tokens = split_to_tokens(lines[i])[-num_columns:]
            if sum(1 for t in tokens if t in registered_raw_units) >= 1:
                units_line_idx = i
                break

    # 5. Build column header names
    headers = split_to_tokens(lines[header_line_idx])[-num_columns:]
    units = split_to_tokens(lines[units_line_idx])[-num_columns:] if units_line_idx is not None else [''] * num_columns

    clean_columns = [f"{h} {u}".strip() if u else h for h, u in zip(headers, units)]

    return {
        'delimiter': delim,
        'skiprows': first_numeric_idx,
        'names': clean_columns,
        'encoding': encoding,
        'file_hints': file_hints
    }


def read_csv_with_config(file_path: str | Path, csv_config: dict) -> pd.DataFrame:
    """Reads raw CSV using extracted configuration dict."""
    return pd.read_csv(
        file_path,
        sep=csv_config['delimiter'],
        skiprows=csv_config['skiprows'],
        names=csv_config['names'],
        header=None,
        encoding=csv_config['encoding'],
    )


def transform_df_to_si(df: pd.DataFrame, registry_name: str) -> pd.DataFrame:
    """Transforms DataFrame columns and numeric values into standardized SI units."""
    si_df = df.copy()
    rename_map = {}
    si_df.dropna(axis=1, how='all', inplace=True)

    _, registry = load_registry_from_toml(registry_name)
    
    for col in si_df.columns:
        if col in registry:
            mapping = registry[col]
            if pd.api.types.is_numeric_dtype(si_df[col]):
                si_df[col] = mapping.convert_fn(si_df[col])
            
            rename_map[col] = mapping.standard_name
        else:
            fallback = f"_{col.lower().replace(' ', '_')}"
            rename_map[col] = fallback
            logger.warning(f"Column '{col}' not found in registry '{registry_name}'. Renamed to '{fallback}'.")

    si_df.rename(columns=rename_map, inplace=True)
    logger.info(f"Renamed columns: {si_df.columns.tolist()}")
    return si_df


def read_csv_to_si(file_path: str | Path, registry_name: str, encoding=None, delim=None) -> pd.DataFrame:
    """Full workflow reader for raw CSV files."""
    csv_config = get_file_config(file_path, registry_name, encoding=encoding, delim=delim)
    df = read_csv_with_config(file_path, csv_config)
    return transform_df_to_si(df, registry_name)


def read_excel_to_si_dict(
    file_path: str | Path,
    registry_name: str,
    sheets: str | list[str] | dict[str, str] | None = None,
    transform: bool = True
) -> dict[str, pd.DataFrame]:
    """Reads Excel workbook into a dictionary of DataFrames, with unit transformation."""
    if not file_exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    path = Path(file_path)
    sheet_param = list(sheets.keys()) if isinstance(sheets, dict) else sheets

    excel_dict = pd.read_excel(path, sheet_name=sheet_param, engine="openpyxl")
    if isinstance(excel_dict, pd.DataFrame):
        excel_dict = {sheets if isinstance(sheets, str) else "Sheet1": excel_dict}

    processed = {}
    for sheet_name, df in excel_dict.items():
        if transform:
            df = transform_df_to_si(df, registry_name=registry_name)
        processed[sheet_name] = df

    return processed