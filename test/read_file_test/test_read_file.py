# %%
import glob
from func import *

# %%


files = [
    "../test_data/BGPDS50-DTDA-1_kic.CSV",  # Ta-electroforce
    #     '../test_data/BGPDS50-DTDA50_cure_060225.txt',
    #     '../test_data/BGPDS50-DTDA-1_dma.txt',
    #     '../test_data/BGPDS-DTDA-1_compression.txt',
    #     '../test_data/dgeba-mda-6_mdsc.txt',
    #     '../test_data/BGPDS50-DTDA50_190C_060225_rheo.csv',   # anton-paar
]
registry_name = "TA_electroforce"
# registry_name = 'anton-paar_rhometer'

for file in files:
    print("=====================")
    print(f"file: {file}")
    # get encoding
    encoding = get_file_encoding(file)
    print(f"encoding = {encoding}")

    # get delimeter
    with open(file, "r", encoding=encoding, errors="ignore") as f:
        lines = f.readlines()

    delim = get_text_delimiter(lines)
    print(f"delim  = {delim}")

    # get config with minimum information
    config = get_file_config(
        file, registry_name
    )  # using the function from previous response
    print(f"config = {config}")

    # read raw csv
    raw_df = read_csv_with_config(file, config)
    print(f"raw_df head")
    print(raw_df.head())

    # transfer to si unit
    si_df = transform_df_to_si(raw_df, registry_name)
    print(f"si_df head")
    print(si_df.head())

    # function in one with minimum information
    df = read_csv_to_si(file, registry_name)


# %%
