# %%
import glob
from func import *
import logging.config
# Get the logger specified in the file
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # Set the logging level to for output to console

# %%


files = [
    "./test_data/BGPDS50-DTDA-1_kic.CSV",  # Ta-electroforce
    #     './test_data/BGPDS50-DTDA50_cure_060225.txt',
    #     './test_data/BGPDS50-DTDA-1_dma.txt',
    #     './test_data/BGPDS-DTDA-1_compression.txt',
    #     './test_data/dgeba-mda-6_mdsc.txt',
    #     './test_data/BGPDS50-DTDA50_190C_060225_rheo.csv',   # anton-paar
]
registry_name = "TA_electroforce"
# registry_name = 'anton-paar_rhometer'

for file in files:
    logger.info("=====================")
    logger.info(f"file: {file}")
    # get encoding
    encoding = get_file_encoding(file)
    logger.info(f"encoding = {encoding}")

    # get delimeter
    with open(file, "r", encoding=encoding, errors="ignore") as f:
        lines = f.readlines()

    delim = get_text_delimiter(lines)
    logger.info(f"delim  = {delim}")

    # get config with minimum information
    config = get_file_config(
        file, registry_name
    )  # using the function from previous response
    logger.info(f"config = {config}")

    # read raw csv
    raw_df = read_csv_with_config(file, config)
    logger.info(f"raw_df head")
    logger.info(raw_df.head())

    # transfer to si unit
    si_df = transform_df_to_si(raw_df, registry_name)
    logger.info(f"si_df head")
    logger.info(si_df.head())

    # function in one with minimum information
    df = read_csv_to_si(file, registry_name)


# %%
import tomllib

# 1. Read and parse the TOML file
with open("./registries/standards.toml", "rb") as f:
    data = tomllib.load(f)

# 2. Dynamically print every category, variable, and key-value pair
for category, variables in data.items():
    print(f"\n=== {category.upper()} ===")
    for var_name, metadata in variables.items():
        print(f"  [{var_name}]")
        for key, val in metadata.items():
                        # Strips the $ sign if present for cleaner console viewing
            clean_val = str(val).strip("$") if isinstance(val, str) else val
            print(f"    {key:<15} : {clean_val}")
# %%


