# %%
from pathlib import Path
import logging
import tomllib

from fileManager import (
    get_file_encoding,
    get_text_delimiter,
    get_file_config,
    read_csv_with_config,
    transform_df_to_si,
    read_csv_to_si,
    load_standards_registry,
)

# Configure centralized logger for testing
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# %%
# Define test files and matching registry names
files_and_registries = [
    ("./test_data/BGPDS50-DTDA-1_kic.CSV", "ta-instruments_mechanics"),
    # ("./test_data/BGPDS50-DTDA50_190C_060225_rheo.csv", "anton-paar_rheology"),
]

for file_path, registry_name in files_and_registries:
    path = Path(file_path)
    if not path.is_file():
        logger.warning(f"Skipping missing test file: {file_path}")
        continue

    logger.info("=" * 60)
    logger.info(f"Processing File: {path.name} | Registry: {registry_name}")
    
    # 1. Inspect low-level file parameters
    encoding = get_file_encoding(path)
    logger.info(f"Detected Encoding: {encoding}")

    with open(path, "r", encoding=encoding, errors="ignore") as f:
        lines = f.readlines()

    delim = get_text_delimiter(lines)
    logger.info(f"Detected Delimiter: {repr(delim)}")

    # 2. Extract configuration
    config = get_file_config(path, registry_name)
    logger.info(f"Extracted CSV Config: {config}")

    # 3. Read raw DataFrame
    raw_df = read_csv_with_config(path, config)
    logger.info("Raw DataFrame Head:")
    logger.info("\n" + str(raw_df.head()))

    # 4. Transform DataFrame to SI units
    si_df = transform_df_to_si(raw_df, registry_name)
    logger.info("Transformed SI DataFrame Head:")
    logger.info("\n" + str(si_df.head()))

    # 5. One-step reader verification
    df_direct = read_csv_to_si(path, registry_name)
    logger.info(f"Direct Reader Output Shape: {df_direct.shape}")


# %%
# Verify and print standard target variables from standards.toml
logger.info("=" * 60)
logger.info("LOADING STANDARDS REGISTRY")

standards_data = load_standards_registry()

for category, variables in standards_data.items():
    print(f"\n=== {category.upper()} ===")
    for var_name, metadata in variables.items():
        print(f"  [{var_name}]")
        for key, val in metadata.items():
            clean_val = str(val).strip("$").replace("\\\\", "\\") if isinstance(val, str) else val
            print(f"    {key:<15} : {clean_val}")

# %%